import numpy as np
import random
import matplotlib.pyplot as plt

class IsingModel2D:
    # 初始化各种参数和状态
    def __init__(self, L, T, J_value=1.0, H=0.0, num_states_for_ising_potts=2, 
                 use_random_J=False, random_J_max=1.0, 
                 use_potts_interaction=False, use_xy_model=False):
        self.L = L # 晶格边长
        if T <= 0: self.T = 1e-8 
        else: self.T = T
        
        self.H_scalar = H # 外场强度，XY模型中假定沿x轴
        self.N = L * L # 晶格总点数
        self.beta = 1.0 / self.T  # 玻尔兹曼因子 beta = 1/(k_B T)，这里 k_B 设为1

        self.use_xy_model = use_xy_model 
        self.use_potts_interaction = use_potts_interaction if not self.use_xy_model else False
        self.use_random_J = use_random_J

        self.model_type_str = "" 
        # 使用连续，矢量 si (XY)模型：
        if self.use_xy_model:
            self.lattice = np.random.uniform(0, 2 * np.pi, size=(L, L)) # 随机初始化XY模型的角度
            self.model_type_str = "XY"
            self.num_states = 0
            self.spin_values = None
        # 使用离散，标量 si (Ising/Potts)模型：
        else:
            self.num_states = int(num_states_for_ising_potts) # si的可能状态数
            if self.num_states <= 1 : 
                 raise ValueError("Ising/Potts模型的自旋状态数必须大于1")
            self.spin_values = np.arange(self.num_states) - (self.num_states - 1) / 2.0 # 生成自旋值范围
            self.lattice = np.random.choice(self.spin_values, size=(L, L)) # 初始化晶格自旋值
            self.model_type_str = "Potts" if self.use_potts_interaction else "Ising"

        j_config_str_part = ""
        if self.use_random_J:
            self.J_max_for_random = abs(random_J_max) if not np.isclose(random_J_max, 0) else 1.0
            # 为每个水平和垂直键生成 [-J_max, J_max] 范围内的随机J值
            self.J_horizontal = np.random.uniform(-self.J_max_for_random, self.J_max_for_random, size=(L, L))
            self.J_vertical = np.random.uniform(-self.J_max_for_random, self.J_max_for_random, size=(L, L))
            j_config_str_part = f"Rand(±{self.J_max_for_random:.2f})"
        else:
            self.J_single = J_value # 单一J值
            j_config_str_part = f"{self.J_single:.2f}"
        
        self.J_display_str = f"{self.model_type_str} J={j_config_str_part}"
        if self.use_xy_model and use_potts_interaction: 
             print("Warning:When using XY model, Potts interaction is ignored.")

        # 初始化用于存储磁化强度M和能量E历史数据的列表
        self.magnetization_history = []
        self.energy_history = []
        self.sweep_history_for_plots = []

    # 清空历史数据列表，在开始新的模拟（或重置）时调用
    def clear_history(self): 
        self.magnetization_history.clear(); self.energy_history.clear(); self.sweep_history_for_plots.clear()

    # 计算Delta_E：某格点自旋从旧状态变为新状态时，系统能量的变化
    def _calculate_delta_E(self, r, c, new_spin_val_or_angle):
        if self.use_xy_model:
            theta_old = self.lattice[r, c]
            theta_new = new_spin_val_or_angle 
            if np.isclose(theta_old, theta_new): return 0.0
        else: 
            s_old = self.lattice[r, c]
            s_new = new_spin_val_or_angle
            if np.isclose(s_old, s_new): return 0.0

        # 获取格点(r,c)的四个最近邻的状态值（使用周期性边界条件）
        s_up = self.lattice[(r - 1) % self.L, c]
        s_down = self.lattice[(r + 1) % self.L, c]
        s_left = self.lattice[r, (c - 1) % self.L]
        s_right = self.lattice[r, (c + 1) % self.L]
        neighbors_states = [s_up, s_down, s_left, s_right]
        
        # 获取连接格点(r,c)与邻居的四个J值
        if self.use_random_J:
            Js_for_bonds = [ self.J_vertical[(r - 1) % self.L, c], self.J_vertical[r, c], 
                             self.J_horizontal[r, (c - 1) % self.L], self.J_horizontal[r, c] ]
        else:
            Js_for_bonds = [self.J_single] * 4

        delta_E_interaction = 0 # 初始化ΔE的相互作用部分
        delta_E_field = 0 # 初始化ΔE的外场部分

        if self.use_xy_model:
            # XY模型能量计算: E_interaction = -J * cos(theta_i - theta_j)
            for i in range(4): # 分别计算四边再相加
                J_bond = Js_for_bonds[i]
                theta_k = neighbors_states[i] 
                delta_E_interaction += -J_bond * (np.cos(theta_new - theta_k) - np.cos(theta_old - theta_k))
            # 外场对XY模型的贡献 (假定外场H沿x轴方向, H_scalar > 0 表示场指向 +x)
            delta_E_field = -self.H_scalar * (np.cos(theta_new) - np.cos(theta_old))
        elif self.use_potts_interaction: 
            # Potts模型能量计算: E_interaction = -J * delta(s_i, s_j)
            for i in range(4): # 分别计算四边再相加
                J_bond = Js_for_bonds[i]
                s_k = neighbors_states[i] 
                term_old = 1.0 if np.isclose(s_old, s_k) else 0.0
                term_new = 1.0 if np.isclose(s_new, s_k) else 0.0
                delta_E_interaction += -J_bond * (term_new - term_old)
            # 外场对Potts模型的贡献
            delta_E_field = -self.H_scalar * (s_new - s_old) 
        else:
            # Ising能量计算: E_interaction = -J * s_i * s_j
            for i in range(4):
                J_bond = Js_for_bonds[i]
                s_k = neighbors_states[i]
                delta_E_interaction += -J_bond * s_k * (s_new - s_old)
            # 外场对Ising的贡献
            delta_E_field = -self.H_scalar * (s_new - s_old)
            
        return delta_E_interaction + delta_E_field # 总能量变化= ΔE_interaction + ΔE_field

    # Metropolis算法的单步更新
    def metropolis_step(self):
        # 随机选一个格点
        r, c = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
        
        if self.use_xy_model:
            # 随机生成一个新的角度
            new_angle = np.random.uniform(0, 2 * np.pi)
            # 计算能量变化
            delta_E = self._calculate_delta_E(r, c, new_angle)
            # 如果能量降低或满足Metropolis准则，则接受新角度
            if delta_E <= 0 or (self.T > 1e-9 and random.random() < np.exp(-self.beta * delta_E)):
                self.lattice[r, c] = new_angle
        else: 
            s_old = self.lattice[r,c]
            if self.num_states <= 1: return 

            # 从除了当前状态以外的其他可能状态中随机选择一个新状态
            possible_new_states = [s for s in self.spin_values if not np.isclose(s, s_old)]
            if not possible_new_states: return 

            # 随机选择一个新的状态
            s_new = random.choice(possible_new_states)
            # 计算能量变化
            delta_E = self._calculate_delta_E(r, c, s_new)
            # 如果能量降低或满足Metropolis准则，则接受新状态
            if delta_E <= 0 or (self.T > 1e-9 and random.random() < np.exp(-self.beta * delta_E)):
                self.lattice[r, c] = s_new

    # 运行nums_sweeps_to_run次sweep
    # 每个sweep包含N个Metropolis单步(sweep:平均每个自旋都被尝试改变一次)
    def run_model_sweeps(self, num_sweeps_to_run): 
        for _ in range(num_sweeps_to_run): [self.metropolis_step() for _ in range(self.N)]

    # 计算系统当前平均的磁化强度
    def calculate_magnetization(self):
        # XY模型：计算沿x轴的平均磁化分量 M_x = <cos(theta_i)>
        if self.use_xy_model:
            return np.mean(np.cos(self.lattice))
        # Ising/Potts模型：计算平均自旋值
        else: 
            return np.sum(self.lattice) / self.N

    # 计算系统当前状态的总能量
    def calculate_total_energy(self):
        energy_J_interaction = 0 # 初始化能量的相互作用部分
        energy_H_field = 0 # 初始化能量的外场部分
        # 遍历每个格点与其右方和下方邻居的相互作用
        for r_idx in range(self.L):
            for c_idx in range(self.L):
                val_ic = self.lattice[r_idx, c_idx] # 当前格点的值
                
                # 获取右方和下方邻居的值（使用周期性边界条件）
                val_right = self.lattice[r_idx, (c_idx + 1) % self.L]
                val_down = self.lattice[(r_idx + 1) % self.L, c_idx]

                # 获取当前格点(r_idx, c_idx)的J值
                J_h = self.J_horizontal[r_idx, c_idx] if self.use_random_J else self.J_single
                J_v = self.J_vertical[r_idx, c_idx] if self.use_random_J else self.J_single
                
                # 计算相互作用能量和外场能量
                if self.use_xy_model:
                    energy_J_interaction += -J_h * np.cos(val_ic - val_right)
                    energy_J_interaction += -J_v * np.cos(val_ic - val_down)
                    energy_H_field += -self.H_scalar * np.cos(val_ic)
                elif self.use_potts_interaction:
                    energy_J_interaction += -J_h * (1.0 if np.isclose(val_ic, val_right) else 0.0)
                    energy_J_interaction += -J_v * (1.0 if np.isclose(val_ic, val_down) else 0.0)
                    energy_H_field += -self.H_scalar * val_ic 
                else: # Ising-like
                    energy_J_interaction += -J_h * val_ic * val_right
                    energy_J_interaction += -J_v * val_ic * val_down
                    energy_H_field += -self.H_scalar * val_ic

        # 返回总能量=相互作用能 + 外场能量 
        return energy_J_interaction + energy_H_field

    # 在给定的Matplotlib Axes对象显示当前晶格状态
    def display_on_ax(self, ax, title_info_dynamic="", cmap_name='viridis'):
        ax.clear() # 清除之前的绘图内容
        
        model_params_title = f"{self.J_display_str}" 
        model_params_title += f" H={self.H_scalar if self.H_scalar is not None else 0.0}"
        if not self.use_xy_model: 
             model_params_title += f" Ns={self.num_states}"
        full_title = f"{model_params_title}\n{title_info_dynamic}"
        ax.set_title(full_title, fontsize=5) 
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0)

        # XY模型：使用quiver图绘制箭头表示自旋方向
        if self.use_xy_model:
            L_plot = self.lattice.shape[0]
            X, Y = np.meshgrid(np.arange(L_plot), np.arange(L_plot))
            Angles = self.lattice
            U = np.cos(Angles)
            V = np.sin(Angles)
            
            # 获取颜色映射对象，Error则使用HSV色图
            try:
                cmap_obj_xy = plt.get_cmap(cmap_name if cmap_name in ['hsv', 'twilight', 'twilight_shifted', 'rainbow', 'turbo'] else 'hsv')
            except ValueError:
                cmap_obj_xy = plt.get_cmap('hsv')
            
            # 将角度归一化到[0,1]以便用于colormap
            norm_angles = (Angles % (2 * np.pi)) / (2 * np.pi) if Angles.size > 0 else np.array([])
            colors_for_quiver = cmap_obj_xy(norm_angles) if norm_angles.size > 0 else 'blue'
            
            # 绘制quiver图
            ax.quiver(X, Y, U, V, color=colors_for_quiver, 
                      scale=L_plot*1.8, width=0.012,
                      headwidth=3, headlength=4, pivot='middle',
                      angles='uv')
            ax.set_xlim([-1, L_plot])
            ax.set_ylim([L_plot, -1]) 
            ax.invert_yaxis() 
            ax.set_aspect('equal', adjustable='box')

        # Ising / Potts 模型: 使用imshow绘制热图
        else: 
            try: cmap_obj = plt.get_cmap(cmap_name)
            except ValueError: cmap_obj = plt.get_cmap('viridis')

            if self.spin_values is None:
                print("Error: spin_values is None, cannot determine vmin/vmax for imshow.")
                return

            vmin_val, vmax_val = self.spin_values.min(), self.spin_values.max()
            if np.isclose(vmin_val, vmax_val): vmin_val -= 0.5; vmax_val += 0.5
            ax.imshow(self.lattice, cmap=cmap_obj, vmin=vmin_val, vmax=vmax_val, interpolation='nearest')