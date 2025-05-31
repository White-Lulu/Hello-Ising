import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import datetime
import time
import os

from ising_model import IsingModel2D
class IsingApp:
    ### ----- 初始化与设置类函数 -----
    def __init__(self, master_root):

        # Tkinter根窗口参数
        self.root = master_root
        self.root.title("Hello Ising 2D ^v^")
        self.root.geometry("1500x575")

        # 线程参数
        self.simulation_thread = None
        self.pause_event = threading.Event(); self.pause_event.set()
        self.stop_event = threading.Event()
        self.current_simulation_id = 0

        ## --- [可更改的默认值] ---

        # 常规模拟参数 
        self.L_val = 70; # 格点数目
        self.H_scalar = 0.0 # 外场强度
        self.H_scalar_gui=self.H_scalar
        self.total_sweeps_to_show = 1000 # 总sweeps
        self.sweeps_per_frame = 2 # 每隔多少sweeps更新一次图像
        self.data_log_interval = 5 # 每隔多少sweeps记录一次M/E数据

        self.models_for_current_sim = {} # 存储当前常规模拟中的模型实例 (键为模型名，值为模型对象)
        self.axes_configs_list = [] # 存储常规模拟中每个温度块的Axes配置字典列表
        self._current_selected_temps_values = [] # 存储常规模拟中选择的温度的列表

        # 批量模拟参数
        self.batch_t_min_var = tk.StringVar(value="0.5")
        self.batch_t_max_var = tk.StringVar(value="3.5")
        self.batch_num_temp_points_var = tk.IntVar(value=10)
        self.batch_sweeps_per_temp_var = tk.StringVar(value="500")
        self.batch_save_snapshots_var = tk.BooleanVar(value=False)
        self.batch_temperatures_scanned = []
        self.batch_final_M_values = []
        self.batch_final_E_values = []
        self.batch_lattice_ax = None
        self.batch_M_vs_T_ax = None
        self.batch_E_vs_T_ax = None
        self.is_batch_mode_active = False
        self.last_batch_model_config = {}

        # GUI设置参数
        self.selected_colormap_var = tk.StringVar(value='viridis') # 默认颜色方案
        self.colormaps_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'binary', 'coolwarm', 'RdBu_r', 'jet', 'turbo', 'rainbow', 'hsv', 'twilight', 'twilight_shifted'] # 可选颜色方案
        self.plot_M_var = tk.BooleanVar(value=False)
        self.plot_E_var = tk.BooleanVar(value=False)
        self.random_J_var = tk.BooleanVar(value=False)
        self.use_potts_model_var = tk.BooleanVar(value=False) 
        self.j_slider_actual_label_text = tk.StringVar(value="J 值 (固定):")
        self.use_xy_model_var = tk.BooleanVar(value=False) 
        self.batch_save_data_var = tk.BooleanVar(value=False)

        self._setup_gui()  # 调用方法来创建和布局所有GUI控件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing) # 注册窗口关闭事件的处理函数
        self._initialize_gui_states() # 初始化GUI控件的初始状态

    def _initialize_gui_states(self):
        self._on_random_j_toggle() 
        self._on_xy_model_toggle() 

    def _setup_gui(self):
        # --- 主框架：左侧滚动控制区，右侧绘图区 ---
        outer_control_container = ttk.Frame(self.root)
        outer_control_container.grid(row=0, column=0, sticky="nsew", padx=(10,0), pady=10)

        # 右侧绘图区
        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # 配置根窗口列权重
        self.root.grid_columnconfigure(0, weight=5) 
        self.root.grid_columnconfigure(1, weight=5)
        self.root.grid_rowconfigure(0, weight=1)

        # 左侧滚动控制区
        canvas_for_controls = tk.Canvas(outer_control_container, borderwidth=0, highlightthickness=0)
        scrollbar_for_controls = ttk.Scrollbar(outer_control_container, orient="vertical", command=canvas_for_controls.yview)
        canvas_for_controls.configure(yscrollcommand=scrollbar_for_controls.set)
        
        scrollbar_for_controls.pack(side="right", fill="y")
        canvas_for_controls.pack(side="left", fill="both", expand=True)

        self.scrollable_content_frame = ttk.Frame(canvas_for_controls, padding="5")
        canvas_for_controls.create_window((0, 0), window=self.scrollable_content_frame, anchor="nw", tags="self.scrollable_content_frame")
        
        def on_mousewheel(event): canvas_for_controls.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter_canvas(event): canvas_for_controls.bind_all("<MouseWheel>", on_mousewheel)
        def on_leave_canvas(event): canvas_for_controls.unbind_all("<MouseWheel>")
        canvas_for_controls.bind('<Enter>', on_enter_canvas); canvas_for_controls.bind('<Leave>', on_leave_canvas)
        
        def update_scroll_region(event): canvas_for_controls.configure(scrollregion=canvas_for_controls.bbox("all"))
        self.scrollable_content_frame.bind("<Configure>", update_scroll_region)
        
        # 两列布局的父框架
        main_control_subframe = ttk.Frame(self.scrollable_content_frame)
        main_control_subframe.pack(fill="x", expand=True) # 使用pack让它填充scrollable_content_frame

        left_column_container = ttk.Frame(main_control_subframe)
        left_column_container.grid(row=0, column=0, sticky="new", padx=(0,5)) # new 而不是 nsew
        right_column_container = ttk.Frame(main_control_subframe)
        right_column_container.grid(row=0, column=1, sticky="new", padx=(5,0)) # new 而不是 nsew
        
        main_control_subframe.grid_columnconfigure(0, weight=1) 
        main_control_subframe.grid_columnconfigure(1, weight=1)

        # 左列控件
        current_row_left = 0 
        frame_j = ttk.LabelFrame(left_column_container, text="耦合常数 (J)", padding=(5,3)); frame_j.grid(row=current_row_left, column=0, pady=3, sticky="ew"); current_row_left += 1
        self.j_slider_label_widget = ttk.Label(frame_j, textvariable=self.j_slider_actual_label_text); self.j_slider_label_widget.pack(anchor="w", padx=3)
        self.j_slider = tk.Scale(frame_j, from_=-5.0, to=5.0, resolution=0.1, orient="horizontal", length=200); self.j_slider.set(1.0); self.j_slider.pack(fill="x", padx=3) # 调整length
        
        frame_h = ttk.LabelFrame(left_column_container, text="外场强度 (H)", padding=(5,3)); frame_h.grid(row=current_row_left, column=0, pady=3, sticky="ew"); current_row_left += 1
        h_label = ttk.Label(frame_h, text="H 值:"); h_label.pack(anchor="w", padx=3)
        self.h_slider = tk.Scale(frame_h, from_=-2.0, to=2.0, resolution=0.05, orient="horizontal", length=200); self.h_slider.set(self.H_scalar_gui); self.h_slider.pack(fill="x", padx=3) # 调整length
        
        model_type_frame = ttk.LabelFrame(left_column_container, text="模型规则", padding=(5,3)); model_type_frame.grid(row=current_row_left, column=0, pady=3, sticky="ew"); current_row_left += 1
        self.random_j_checkbox = ttk.Checkbutton(model_type_frame, text="随机J<自旋玻璃> (J∈[-Jmax,Jmax])", variable=self.random_J_var, command=self._on_random_j_toggle); self.random_j_checkbox.pack(anchor="w")
        self.potts_checkbox = ttk.Checkbutton(model_type_frame, text="Potts 模型 (-Jδsij)", variable=self.use_potts_model_var, command=self._on_potts_model_toggle); self.potts_checkbox.pack(anchor="w")
        self.xy_checkbox = ttk.Checkbutton(model_type_frame, text="XY 模型 (-Jcos(Δθ))", variable=self.use_xy_model_var, command=self._on_xy_model_toggle); self.xy_checkbox.pack(anchor="w")
        
        frame_ns = ttk.LabelFrame(left_column_container, text="自旋/状态配置", padding=(5,3)); frame_ns.grid(row=current_row_left, column=0, pady=3, sticky="ew"); current_row_left += 1
        self.ns_slider_label = ttk.Label(frame_ns, text="状态数 s (Ising):"); self.ns_slider_label.pack(anchor="w", padx=3)
        self.ns_slider = tk.Scale(frame_ns, from_=2, to=8, resolution=1, orient="horizontal", length=200); self.ns_slider.set(2); self.ns_slider.pack(fill="x", padx=3) # 调整length
        
        frame_temps = ttk.LabelFrame(left_column_container, text="常规模拟温度 (T)", padding=(5,3)); frame_temps.grid(row=current_row_left, column=0, pady=3, sticky="ew"); current_row_left += 1
        temp_slider_frame = ttk.Frame(frame_temps); temp_slider_frame.pack(fill="x")
        ttk.Label(temp_slider_frame, text="选择T:").pack(side="left", padx=3)
        self.temp_slider = tk.Scale(temp_slider_frame, from_=0.1, to=5.0, resolution=0.05, orient="horizontal", length=140); self.temp_slider.set(2.269); self.temp_slider.pack(side="left") # 调整length
        listbox_frame = ttk.Frame(frame_temps); listbox_frame.pack(fill="x", pady=3)
        self.added_temps_listbox = tk.Listbox(listbox_frame, height=2, width=8, exportselection=False); self.added_temps_listbox.pack(side="left", padx=(0,3)) # 调整宽度
        temp_buttons_frame = ttk.Frame(listbox_frame); temp_buttons_frame.pack(side="left")
        self.add_temp_button = ttk.Button(temp_buttons_frame, text="添加T", command=self._add_temperature, width=6); self.add_temp_button.pack(pady=1) # 调整宽度和文本
        self.remove_temp_button = ttk.Button(temp_buttons_frame, text="移除T", command=self._remove_selected_temperature, width=6); self.remove_temp_button.pack(pady=1) # 调整宽度和文本

        # 右列控件
        current_row_right = 0
        plot_options_frame = ttk.LabelFrame(right_column_container, text="常规模拟绘图", padding=(5,3)); plot_options_frame.grid(row=current_row_right, column=0, pady=3, sticky="ew"); current_row_right += 1
        self.plot_M_var_checkbox = ttk.Checkbutton(plot_options_frame, text="M vs. Sweeps", variable=self.plot_M_var); self.plot_M_var_checkbox.pack(anchor="w")
        self.plot_E_var_checkbox = ttk.Checkbutton(plot_options_frame, text="E vs. Sweeps", variable=self.plot_E_var); self.plot_E_var_checkbox.pack(anchor="w")
        
        frame_colormap = ttk.LabelFrame(right_column_container, text="颜色方案", padding=(5,3)); frame_colormap.grid(row=current_row_right, column=0, pady=5, sticky="ew"); current_row_right += 1
        self.colormap_combobox_label = ttk.Label(frame_colormap, text="热图/箭头颜色:"); self.colormap_combobox_label.pack(side="left", padx=(3,2))
        self.colormap_combobox = ttk.Combobox(frame_colormap, textvariable=self.selected_colormap_var, values=self.colormaps_list, state="readonly", width=12); self.colormap_combobox.set('viridis'); self.colormap_combobox.pack(pady=3, padx=2, side="left") # 调整宽度
        
        sim_control_frame = ttk.LabelFrame(right_column_container, text="常规模拟", padding=(5,3)); sim_control_frame.grid(row=current_row_right, column=0, pady=10, sticky="ew"); current_row_right += 1
        self.start_button = ttk.Button(sim_control_frame, text="开始常规模拟", command=self._start_or_reset_simulation); self.start_button.pack(fill="x", pady=2)
        
        batch_frame = ttk.LabelFrame(right_column_container, text="批量实验", padding=(5,5)); batch_frame.grid(row=current_row_right, column=0, pady=6, sticky="ew"); current_row_right += 1
        temp_range_f = ttk.Frame(batch_frame); temp_range_f.pack(fill="x", pady=2)
        ttk.Label(temp_range_f, text="温度区间 T: ").pack(side="left")
        self.t_min_batch_entry = ttk.Entry(temp_range_f, textvariable=self.batch_t_min_var, width=5); self.t_min_batch_entry.pack(side="left", padx=1)
        ttk.Label(temp_range_f, text="~").pack(side="left")
        self.t_max_batch_entry = ttk.Entry(temp_range_f, textvariable=self.batch_t_max_var, width=5); self.t_max_batch_entry.pack(side="left", padx=1)
        num_points_f = ttk.Frame(batch_frame); num_points_f.pack(fill="x", pady=2)
        ttk.Label(num_points_f, text="温度点数量:").pack(side="left")
        self.num_temp_points_batch_slider = tk.Scale(num_points_f, variable=self.batch_num_temp_points_var, from_=2, to=50, resolution=1, orient="horizontal", length=150); self.num_temp_points_batch_slider.set(10); self.num_temp_points_batch_slider.pack(side="left", padx=3)
        sweeps_batch_f = ttk.Frame(batch_frame); sweeps_batch_f.pack(fill="x", pady=2)
        ttk.Label(sweeps_batch_f, text="每温度点Sweeps数:").pack(side="left")
        self.sweeps_per_temp_batch_entry = ttk.Entry(sweeps_batch_f, textvariable=self.batch_sweeps_per_temp_var, width=7); self.sweeps_per_temp_batch_entry.pack(side="left", padx=2)
        save_options_f = ttk.Frame(batch_frame); save_options_f.pack(fill="x", pady=2)
        self.save_snapshots_batch_checkbox = ttk.Checkbutton(save_options_f, text="保存快照", variable=self.batch_save_snapshots_var); self.save_snapshots_batch_checkbox.pack(side="left", anchor="w", padx=(0,10))
        self.save_data_batch_checkbox = ttk.Checkbutton(save_options_f, text="保存数据(T,M,E)", variable=self.batch_save_data_var); self.save_data_batch_checkbox.pack(side="left", anchor="w")
        self.start_batch_button = ttk.Button(batch_frame, text="开始批量实验", command=self._start_batch_experiment_action); self.start_batch_button.pack(fill="x", pady=5)

        global_sim_control_frame = ttk.LabelFrame(right_column_container, text="全局控制", padding=(5,5)); global_sim_control_frame.grid(row=current_row_right, column=0, pady=5, sticky="ew"); current_row_right += 1
        self.pause_button = ttk.Button(global_sim_control_frame, text="暂停", command=self._toggle_pause_simulation, state="disabled"); self.pause_button.pack(fill="x", pady=3)
        self.reset_button = ttk.Button(global_sim_control_frame, text="重置当前模拟", command=self._reset_current_simulation_action, state="disabled"); self.reset_button.pack(fill="x", pady=3)

        # 右侧绘图区
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame) # plot_frame 现在是 grid 的一部分
        self.canvas_widget = self.canvas.get_tk_widget(); self.canvas_widget.pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame); toolbar.update()
        
        self._initialize_gui_states()

    ### ----- GUI状态切换与控制函数 -----

    def _on_random_j_toggle(self):
        if self.random_J_var.get(): self.j_slider_actual_label_text.set("J 幅度上限 (随机 J):")
        else: self.j_slider_actual_label_text.set("J 值 (固定):")

    def _on_xy_model_toggle(self):
        is_xy = self.use_xy_model_var.get()
        common_label_update = lambda text: self.colormap_combobox_label.config(text=text) if hasattr(self, 'colormap_combobox_label') else None
        if is_xy:
            self.ns_slider.config(state="disabled"); self.ns_slider_label.config(text="状态数 s (XY禁用):", foreground="gray")
            if hasattr(self, 'potts_checkbox'): self.use_potts_model_var.set(False); self.potts_checkbox.config(state="disabled")
            self.colormap_combobox.config(state="disabled")
        else: 
            self.ns_slider.config(state="normal")
            if hasattr(self, 'potts_checkbox'): self.potts_checkbox.config(state="normal")
            self._on_potts_model_toggle()
            common_label_update("热图颜色方案:")
        self._on_random_j_toggle() 
    
    def _on_potts_model_toggle(self):
        is_potts = self.use_potts_model_var.get()
        common_label_update = lambda text: self.colormap_combobox_label.config(text=text) if hasattr(self, 'colormap_combobox_label') else None
        if not self.use_xy_model_var.get(): 
            if is_potts:
                if hasattr(self, 'xy_checkbox'): self.xy_checkbox.config(state="disabled") 
                self.ns_slider.config(state="normal"); self.ns_slider_label.config(text="状态数 q (Potts):", foreground="black")
            else: 
                if hasattr(self, 'xy_checkbox'): self.xy_checkbox.config(state="normal")
                self.ns_slider.config(state="normal"); self.ns_slider_label.config(text="状态数 s (Ising):", foreground="black")
        self._on_random_j_toggle()

    def _set_controls_state(self, state_str):
        widgets_to_toggle = [
            self.j_slider, self.h_slider, self.ns_slider, self.temp_slider,
            self.random_j_checkbox, self.potts_checkbox, self.xy_checkbox,
            self.plot_M_var_checkbox, self.plot_E_var_checkbox, 
            self.colormap_combobox,
            self.add_temp_button, self.remove_temp_button,
            self.start_button,
            self.save_data_batch_checkbox,
        ]
        for widget in widgets_to_toggle:
            if hasattr(widget, 'config'): widget.config(state=state_str)
        
        self.start_batch_button.config(state="normal" if state_str == "disabled" else "disabled")

    ### ----- 常规模拟核心函数 -----

    def _start_or_reset_simulation(self, is_reset=False):
        self.current_simulation_id += 1
        local_sim_id = self.current_simulation_id

        if self.simulation_thread and self.simulation_thread.is_alive():
            print(f"尝试停止旧模拟任务{local_sim_id -1}")
            self.stop_event.set(); self.pause_event.set()
            self.simulation_thread.join(timeout=1.5)
        
        self.stop_event.clear(); self.pause_event.set()
        self.pause_button.config(text="暂停", state="normal")
        self.reset_button.config(state="normal")
        self.start_button.config(text="常规模拟运行中...^v^",state="disabled")
        self.start_batch_button.config(text="开始批量实验")

        j_val_from_slider = self.j_slider.get()
        h_val_from_slider = self.h_slider.get()
        num_s_val_ising_potts = int(self.ns_slider.get()) if not self.use_xy_model_var.get() else 2 
        current_temperatures = list(self._current_selected_temps_values)
        self.plot_M_enabled = self.plot_M_var.get()
        self.plot_E_enabled = self.plot_E_var.get()
        use_random_j_flag = self.random_J_var.get()
        use_potts_interaction_flag = self.use_potts_model_var.get()
        use_xy_model_flag = self.use_xy_model_var.get()

        if not current_temperatures:
            messagebox.showerror("错误", "请至少添加一个温度"); self.pause_button.config(state="disabled"); return
        
        self.fig.clear()
        self.axes_configs_list.clear()
        num_temps = len(current_temperatures)
        if num_temps == 0: self.canvas.draw_idle(); return

        gs_main = self.fig.add_gridspec(1, num_temps, wspace=0.35 if num_temps > 1 else 0.05, hspace=0.3) 
        for i in range(num_temps): 
            axes_config = {}; has_side_plots = self.plot_M_enabled or self.plot_E_enabled
            if has_side_plots:
                lattice_width_ratio = 1.5 if use_xy_model_flag else 1 
                gs_temp_block = gs_main[0, i].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.35) 
                ax_lattice = self.fig.add_subplot(gs_temp_block[0, 0]); 
                if not use_xy_model_flag : ax_lattice.set_aspect('equal', adjustable='box')
                axes_config['lattice'] = ax_lattice
                if self.plot_M_enabled and self.plot_E_enabled:
                    gs_me_sub_block = gs_temp_block[0, 1].subgridspec(4, 1, hspace=0.5, height_ratios=[1, 2, 2, 1]) 
                    axes_config['M'] = self.fig.add_subplot(gs_me_sub_block[1, 0])
                    axes_config['E'] = self.fig.add_subplot(gs_me_sub_block[2, 0])
                elif self.plot_M_enabled: axes_config['M'] = self.fig.add_subplot(gs_temp_block[0, 1])
                elif self.plot_E_enabled: axes_config['E'] = self.fig.add_subplot(gs_temp_block[0, 1])
            else: 
                ax_lattice = self.fig.add_subplot(gs_main[0, i]); 
                if not use_xy_model_flag : ax_lattice.set_aspect('equal', adjustable='box') # Only for imshow
                axes_config['lattice'] = ax_lattice
            self.axes_configs_list.append(axes_config)
        self.canvas.draw_idle()

        self.models_for_current_sim.clear()
        for temp_val in current_temperatures:
            model_key = f"SimID{local_sim_id}_T{temp_val:.3f}_J_Ns{num_s_val_ising_potts}_XY{use_xy_model_flag}"
            model = IsingModel2D(L=self.L_val, T=temp_val, 
                                 J_value=j_val_from_slider, 
                                 H=h_val_from_slider,
                                 num_states_for_ising_potts=num_s_val_ising_potts,
                                 use_random_J=use_random_j_flag, 
                                 random_J_max=j_val_from_slider,
                                 use_potts_interaction=use_potts_interaction_flag,
                                 use_xy_model=use_xy_model_flag) 
            model.clear_history()
            self.models_for_current_sim[model_key] = model
        
        print(f"[任务{local_sim_id}] J:{j_val_from_slider}, Ns:{num_s_val_ising_potts}, Temps:{current_temperatures}, RandJ:{use_random_j_flag}, Potts:{use_potts_interaction_flag}, XY:{use_xy_model_flag}")
        
        self.simulation_thread = threading.Thread(target=self._simulation_loop_worker,
            args=(local_sim_id, list(self.models_for_current_sim.values())), daemon=True)
        self.simulation_thread.start()

    def _simulation_loop_worker(self, sim_id, models_list): 
        """模拟循环的工作线程"""
        print(f"----- 常规模拟 [任务{sim_id}] 启动 -----")
        for current_sweep_num in range(1, self.total_sweeps_to_show + 1):
            if self.stop_event.is_set() or sim_id != self.current_simulation_id:
                print(f"[任务{sim_id}] 停止或过时")
                break
            self.pause_event.wait()

            plot_update_needed = (current_sweep_num % self.sweeps_per_frame == 0 or
                                  current_sweep_num == self.total_sweeps_to_show)
            log_data_this_sweep = (current_sweep_num % self.data_log_interval == 0 or
                                   current_sweep_num == 1 or 
                                   current_sweep_num == self.total_sweeps_to_show)

            current_plot_package = []

            for i, model in enumerate(models_list):
                model.run_model_sweeps(1)

                if log_data_this_sweep:
                    if self.plot_M_enabled: model.magnetization_history.append(model.calculate_magnetization())
                    if self.plot_E_enabled: model.energy_history.append(model.calculate_total_energy())
                    if self.plot_M_enabled or self.plot_E_enabled: model.sweep_history_for_plots.append(current_sweep_num)
                
                if plot_update_needed:
                    mag_display = model.calculate_magnetization()
                    energy_display = model.calculate_total_energy()
                    title_info = f"M={mag_display:.3f}, E={energy_display:.1f}"
                    current_plot_package.append({
                        'ax_config_index': i, 
                        'model_lattice_copy': model.lattice.copy(), 
                        'title_info': title_info, 
                        'model_T': model.T, 
                        'model_J_display_str': model.J_display_str, 
                        'model_Ns_display': "XY" if model.use_xy_model else model.num_states,
                        'model_is_xy': model.use_xy_model, 
                        'model_spin_values': model.spin_values if not model.use_xy_model else None,
                        'M_history': list(model.magnetization_history) if self.plot_M_enabled else None,
                        'E_history': list(model.energy_history) if self.plot_E_enabled else None,
                        'sweep_history': list(model.sweep_history_for_plots) if (self.plot_M_enabled or self.plot_E_enabled) else None,
                    })
            
            if plot_update_needed and current_plot_package:
                suptitle = f"{model.model_type_str} Model - Sweeps: {current_sweep_num}"
                self.root.after(0, self._update_plots_in_gui_thread, sim_id, current_plot_package, suptitle)

            if current_sweep_num % 50 == 0: print(f"Sweep {current_sweep_num}/{self.total_sweeps_to_show} √")
            time.sleep(0.0001)

        if not self.stop_event.is_set() and sim_id == self.current_simulation_id:
            is_last_frame_drawn = (self.total_sweeps_to_show % self.sweeps_per_frame == 0)
            if not is_last_frame_drawn or (self.total_sweeps_to_show % self.data_log_interval !=0 and (self.plot_M_enabled or self.plot_E_enabled)):
                final_plot_package = []
                for i, model in enumerate(models_list):
                    # Ensure final data point is logged if not already
                    if not (model.sweep_history_for_plots and model.sweep_history_for_plots[-1] == self.total_sweeps_to_show):
                        if self.plot_M_enabled: model.magnetization_history.append(model.calculate_magnetization())
                        if self.plot_E_enabled: model.energy_history.append(model.calculate_total_energy())
                        if (self.plot_M_enabled or self.plot_E_enabled): model.sweep_history_for_plots.append(self.total_sweeps_to_show)

                    mag_display = model.calculate_magnetization()
                    energy_display = model.calculate_total_energy()
                    title_info = f"M={mag_display:.3f}, E={energy_display:.3f}"
                    final_plot_package.append({
                        'ax_config_index': i, 'model_lattice_copy': model.lattice.copy(), 'title_info': title_info,
                        'model_T': model.T, 'model_J_display_str': model.J_display_str, 
                        'model_Ns_display': "XY" if model.use_xy_model else model.num_states,
                        'model_is_xy': model.use_xy_model,
                        'model_spin_values': model.spin_values if not model.use_xy_model else None,
                        'M_history': list(model.magnetization_history) if self.plot_M_enabled else None,
                        'E_history': list(model.energy_history) if self.plot_E_enabled else None,
                        'sweep_history': list(model.sweep_history_for_plots) if (self.plot_M_enabled or self.plot_E_enabled) else None,
                    })
                if final_plot_package:
                    suptitle_final = f"Ising/XY/Potts Model (L={self.L_val}) - Final State (Sim ID: {sim_id})"
                    self.root.after(0, self._update_plots_in_gui_thread, sim_id, final_plot_package, suptitle_final)
        if sim_id == self.current_simulation_id: self.root.after(0, lambda: self.pause_button.config(state="disabled"))
        print(f"----- 常规模拟 [任务{sim_id}] 已完成 ^v^ -----")

    def _update_plots_in_gui_thread(self, sim_id_from_worker, plot_data_list, suptitle_str): 
        if not self.root.winfo_exists() or sim_id_from_worker != self.current_simulation_id: return
        if not self.axes_configs_list and plot_data_list : print("警告: axes_configs_list 为空。"); return 

        current_cmap_name_from_gui = self.selected_colormap_var.get()

        for data_item in plot_data_list:
            ax_cfg_idx = data_item['ax_config_index']
            if ax_cfg_idx >= len(self.axes_configs_list): print(f"错误：ax_config_index {ax_cfg_idx} 超出范围。"); continue
            axes_config = self.axes_configs_list[ax_cfg_idx]

            if 'lattice' in axes_config:
                ax_lattice = axes_config['lattice']
                ax_lattice.clear() 

                model_params_title = f"T={data_item['model_T']}, {data_item['model_J_display_str']}"
                ns_display_val = data_item['model_Ns_display']
                model_params_title += f", Ns={ns_display_val}"
                model_params_title += f" H={self.H_scalar if self.H_scalar is not None else 0.0}"
                
                full_title = f"{model_params_title}\n{data_item['title_info']}"
                ax_lattice.set_title(full_title, fontsize=7) 
                ax_lattice.set_xticks([]); ax_lattice.set_yticks([])
                ax_lattice.margins(0)


                if data_item['model_is_xy']:
                    lattice_angles = data_item['model_lattice_copy']
                    L_plot = lattice_angles.shape[0]
                    X, Y = np.meshgrid(np.arange(L_plot), np.arange(L_plot))
                    U = np.cos(lattice_angles)
                    V = np.sin(lattice_angles)
                    
                    try:
                        cmap_obj_xy = plt.get_cmap(current_cmap_name_from_gui if current_cmap_name_from_gui in 
                                                ['hsv', 'twilight', 'twilight_shifted', 'rainbow', 'turbo'] else 'hsv')
                    except ValueError:
                        cmap_obj_xy = plt.get_cmap('hsv')
                    
                    norm_angles = (lattice_angles % (2 * np.pi)) / (2 * np.pi)
                    colors_for_quiver = cmap_obj_xy(norm_angles)
                    colors_for_quiver = colors_for_quiver.reshape(-1, 4)
                    
                    ax_lattice.quiver(X, Y, U, V, 
                                    color=colors_for_quiver,
                                    scale=L_plot*1.8, 
                                    width=0.012,
                                    headwidth=3, 
                                    headlength=4, 
                                    pivot='middle',
                                    angles='uv')
                    
                    ax_lattice.set_xlim([-0.5, L_plot-0.5])
                    ax_lattice.set_ylim([L_plot-0.5, -0.5])
                    ax_lattice.set_aspect('equal', adjustable='box')
                else:
                    try: cmap_obj = plt.get_cmap(current_cmap_name_from_gui)
                    except ValueError: cmap_obj = plt.get_cmap('viridis')
                    spin_vals_for_plot = data_item['model_spin_values']
                    if spin_vals_for_plot is None :
                        print("错误：使用Ising/Potts模型但spin_values为None")
                        continue 
                    vmin_val, vmax_val = spin_vals_for_plot.min(), spin_vals_for_plot.max()
                    if np.isclose(vmin_val, vmax_val): vmin_val -= 0.5; vmax_val += 0.5
                    ax_lattice.imshow(data_item['model_lattice_copy'], cmap=cmap_obj, 
                                      vmin=vmin_val, vmax=vmax_val, interpolation='nearest')
            
            if self.plot_M_enabled and 'M' in axes_config and data_item['M_history'] is not None:
                ax_M = axes_config['M']; ax_M.clear()
                if data_item['sweep_history'] and data_item['M_history']: ax_M.plot(data_item['sweep_history'], data_item['M_history'], marker='.', linestyle='-', markersize=2, color='blue')
                ax_M.set_xlabel("Sweeps", fontsize=6); ax_M.set_ylabel("M", fontsize=6)
                ax_M.tick_params(axis='both', which='major', labelsize=6); ax_M.grid(True, linestyle=':', alpha=0.6)
            if self.plot_E_enabled and 'E' in axes_config and data_item['E_history'] is not None:
                ax_E = axes_config['E']; ax_E.clear()
                if data_item['sweep_history'] and data_item['E_history']: ax_E.plot(data_item['sweep_history'], data_item['E_history'], marker='.', linestyle='-', markersize=2, color='red')
                ax_E.set_xlabel("Sweeps", fontsize=6); ax_E.set_ylabel("E", fontsize=6)
                ax_E.tick_params(axis='both', which='major', labelsize=6); ax_E.grid(True, linestyle=':', alpha=0.6)

        self.fig.suptitle(suptitle_str, fontsize=12)
        
        self.canvas.draw_idle()

    ### ----- 温度处理函数 -----

    def _update_temps_listbox(self):
        self.added_temps_listbox.delete(0, tk.END)
        self._current_selected_temps_values.sort()
        for t in self._current_selected_temps_values: self.added_temps_listbox.insert(tk.END, f"{t:.3f}")

    def _add_temperature(self):
        temp_val = self.temp_slider.get()
        if not any(np.isclose(temp_val, t) for t in self._current_selected_temps_values):
            self._current_selected_temps_values.append(temp_val); self._update_temps_listbox()
        else: messagebox.showinfo("提示", f"温度 {temp_val:.3f} 已添加")

    def _remove_selected_temperature(self):
        sel_indices = self.added_temps_listbox.curselection()
        if not sel_indices: return
        for i in sorted(sel_indices, reverse=True): del self._current_selected_temps_values[i]
        self._update_temps_listbox()

    

    ### ----- 批量实验核心函数 -----

    def _start_batch_experiment_action(self):
        self.current_simulation_id += 1
        local_sim_id = self.current_simulation_id
        self.is_batch_mode_active = True # 设置批量模式标志

        if self.simulation_thread and self.simulation_thread.is_alive():
            print(f"尝试停止旧任务{local_sim_id -1},以开始批量实验...")
            self.stop_event.set(); self.pause_event.set()
            self.simulation_thread.join(timeout=1.5)
        
        self.stop_event.clear(); self.pause_event.set()
        self._set_controls_state("disabled")
        self.pause_button.config(text="暂停批量", state="normal")
        self.reset_button.config(text="停止批量", state="normal")
        self.start_batch_button.config(text="批量实验运行中...OoO", state="disabled")
        self.start_button.config(text="开始常规模拟")

        try:
            t_min = float(self.batch_t_min_var.get())
            t_max = float(self.batch_t_max_var.get())
            num_points = self.batch_num_temp_points_var.get()
            self.sweeps_per_temperature_in_batch = int(self.batch_sweeps_per_temp_var.get())
            if t_min >= t_max or num_points < 2 or self.sweeps_per_temperature_in_batch <=0 :
                messagebox.showerror("Error", "批量实验参数无效 \nTmin < Tmax, 点数 >= 2, Sweeps > 0")
                self._set_controls_state("normal"); self.is_batch_mode_active = False; self.start_batch_button.config(state="normal"); return
        except ValueError:
            messagebox.showerror("Error", "批量实验的温度或Sweeps数必须是有效数字")
            self._set_controls_state("normal"); self.is_batch_mode_active = False; self.start_batch_button.config(state="normal"); return

        temp_points = np.linspace(t_max, t_min, num_points) # 从高温到低温

        # 清空之前批量实验的数据
        self.batch_temperatures_scanned.clear()
        self.batch_final_M_values.clear()
        self.batch_final_E_values.clear()

        # 获取其他固定参数
        j_val = self.j_slider.get()
        h_val = self.h_slider.get()
        num_s_val_ip = int(self.ns_slider.get()) if not self.use_xy_model_var.get() else 2
        use_rand_j = self.random_J_var.get()
        use_potts = self.use_potts_model_var.get()
        use_xy = self.use_xy_model_var.get()
        self.current_colormap_name = self.selected_colormap_var.get() # 在批量实验开始时固定colormap
        
        self.last_batch_model_config = { 'L': self.L_val, 'J_value': j_val, 'H': h_val, 
            'num_states_for_ising_potts': num_s_val_ip, 'use_random_J': use_rand_j, 'random_J_max': j_val, 
            'use_potts_interaction': use_potts, 'use_xy_model': use_xy }
        
        model_config_params = {
            'L': self.L_val, 'J_value': j_val, 'H': h_val, 
            'num_states_for_ising_potts': num_s_val_ip,
            'use_random_J': use_rand_j, 'random_J_max': j_val,
            'use_potts_interaction': use_potts, 'use_xy_model': use_xy
        }

        # 为批量实验设置绘图区域 (1个热图 + M(T) + E(T))
        self.fig.clear()
        self.axes_configs_list.clear() # 此列表不用于批量模式

        # 检查是否需要绘制 M/E 曲线
        has_side_plots = self.plot_M_var.get() or self.plot_E_var.get()

        if has_side_plots:
            gs_batch = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.35)
            gs_left_col = gs_batch[0, 0].subgridspec(1, 1)
            self.batch_lattice_ax = self.fig.add_subplot(gs_left_col[0, 0])
            if not use_xy: self.batch_lattice_ax.set_aspect('equal', adjustable='box')

            self.batch_M_vs_T_ax = None
            self.batch_E_vs_T_ax = None

            if self.plot_M_var.get() and self.plot_E_var.get():
                gs_right_col = gs_batch[0, 1].subgridspec(4, 1, hspace=0.5, height_ratios=[1, 2, 2, 1])
                self.batch_M_vs_T_ax = self.fig.add_subplot(gs_right_col[1, 0])
                self.batch_E_vs_T_ax = self.fig.add_subplot(gs_right_col[2, 0])
            elif self.plot_M_var.get():
                self.batch_M_vs_T_ax = self.fig.add_subplot(gs_batch[0, 1])
            elif self.plot_E_var.get():
                self.batch_E_vs_T_ax = self.fig.add_subplot(gs_batch[0, 1])
        
        else:
            self.batch_lattice_ax = self.fig.add_subplot(111)
            self.batch_M_vs_T_ax = None
            self.batch_E_vs_T_ax = None

        self.canvas.draw_idle()
        print(f"----- 批量模拟 [任务 {local_sim_id}] 温度: {temp_points[-1]} ~ {temp_points[0]} 已启动 -----")

        self.simulation_thread = threading.Thread(
            target=self._batch_simulation_loop_worker,
            args=(local_sim_id, temp_points, model_config_params),
            daemon=True)
        self.simulation_thread.start()

    def _batch_simulation_loop_worker(self, sim_id, temp_points_list, model_config):
        """批量实验的工作线程"""
        save_snapshots = self.batch_save_snapshots_var.get()
        snapshot_dir = "./logs_picture"
        if save_snapshots and not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        for temp_idx, current_T in enumerate(temp_points_list):
            if self.stop_event.is_set() or sim_id != self.current_simulation_id:
                print(f"[任务{sim_id}] 已停止")
                break
            
            print(f"T = {current_T:.3f} ({temp_idx+1}/{len(temp_points_list)})")
            
            # 为当前温度创建模型实例
            current_model = IsingModel2D(T=current_T, **model_config) # 使用解包传递通用参数
            
            # 运行指定数量的sweeps
            for sweep_num_at_this_temp in range(1, self.sweeps_per_temperature_in_batch + 1):
                if self.stop_event.is_set() or sim_id != self.current_simulation_id: break
                self.pause_event.wait()
                current_model.run_model_sweeps(1)
                if sweep_num_at_this_temp % 100 == 0:
                    print(f"Sweep {sweep_num_at_this_temp}/{self.sweeps_per_temperature_in_batch}")
            
            if self.stop_event.is_set() or sim_id != self.current_simulation_id: break # 再次检查

            # 收集此温度点的最终数据
            final_M = current_model.calculate_magnetization()
            final_E = current_model.calculate_total_energy()
            self.batch_temperatures_scanned.append(current_T)
            self.batch_final_M_values.append(final_M)
            self.batch_final_E_values.append(final_E)

            # 准备GUI更新数据包
            plot_package = {
                'current_T_lattice_copy': current_model.lattice.copy(),
                'current_T_model_params': { # 传递模型固有参数
                    'T': current_model.T, 'J_display_str': current_model.J_display_str, 
                    'H_scalar': current_model.H_scalar,
                    'Ns_display': "XY" if current_model.use_xy_model else current_model.num_states,
                    'is_xy': current_model.use_xy_model,
                    'spin_values': current_model.spin_values
                },
                'batch_T_history': list(self.batch_temperatures_scanned),
                'batch_M_history': list(self.batch_final_M_values),
                'batch_E_history': list(self.batch_final_E_values),
            }
            suptitle = f"{current_model.model_type_str}: T={current_T:.3f} (Process : {temp_idx+1}/{len(temp_points_list)})"
            self.root.after(0, self._update_batch_plots_in_gui_thread, sim_id, plot_package, suptitle)

            if save_snapshots:
                self._save_snapshot(sim_id, current_T, self.sweeps_per_temperature_in_batch, snapshot_dir, self.last_batch_model_config)

            time.sleep(0.01) # 避免线程跑太快导致GUI卡顿

        # 批量实验结束后的清理
        self.root.after(0, self._finish_batch_mode, sim_id)

    def _update_batch_plots_in_gui_thread(self, sim_id_from_worker, plot_package, suptitle_str):
        """在GUI线程中更新批量实验的绘图"""

        if not self.root.winfo_exists() or sim_id_from_worker != self.current_simulation_id:
            return

        current_cmap = self.selected_colormap_var.get()
        
        # 更新左侧热图/箭头图
        if self.batch_lattice_ax:
            self.batch_lattice_ax.clear()
            model_p = plot_package['current_T_model_params']
            title_str = f"{model_p['J_display_str']}"
            if not model_p['is_xy']: title_str += f", Ns={model_p['Ns_display']}"
            title_str += f", H={model_p['H_scalar']:.2f}"
            mag_display = plot_package['batch_M_history'][-1] if plot_package['batch_M_history'] else 0.0
            energy_display = plot_package['batch_E_history'][-1] if plot_package['batch_E_history'] else 0.0
            title_str += f"\nM={mag_display:.3f}, E={energy_display:.1f}"
            self.batch_lattice_ax.set_title(title_str, fontsize=7)
            self.batch_lattice_ax.set_xticks([]); self.batch_lattice_ax.set_yticks([])
            self.batch_lattice_ax.margins(0)

            if model_p['is_xy']:
                angles = plot_package['current_T_lattice_copy']
                L = angles.shape[0]
                X, Y = np.meshgrid(np.arange(L), np.arange(L))
                U, V = np.cos(angles), np.sin(angles)
                
                try:
                    cmap_obj = plt.get_cmap(current_cmap if current_cmap in 
                        ['hsv', 'twilight', 'twilight_shifted', 'rainbow', 'turbo'] else 'hsv')
                except ValueError:
                    cmap_obj = plt.get_cmap('hsv')
                
                norm_angles = (angles % (2 * np.pi)) / (2 * np.pi)
                colors_for_quiver = cmap_obj(norm_angles)
                colors_for_quiver = colors_for_quiver.reshape(-1, 4)
                
                self.batch_lattice_ax.quiver(X, Y, U, V, 
                                            color=colors_for_quiver,
                                            scale=L*1.8,
                                            width=0.012,
                                            headwidth=3,
                                            headlength=4,
                                            pivot='middle',
                                            angles='uv')
                self.batch_lattice_ax.set_xlim([-0.5, L-0.5])
                self.batch_lattice_ax.set_ylim([L-0.5, -0.5])
                self.batch_lattice_ax.set_aspect('equal', adjustable='box')
            else: # Ising/Potts
                try: cmap_obj = plt.get_cmap(current_cmap)
                except: cmap_obj = plt.get_cmap('viridis')
                spin_vals = model_p['spin_values']
                if spin_vals is None: return
                vmin,vmax = spin_vals.min(), spin_vals.max()
                if np.isclose(vmin,vmax): vmin-=0.5; vmax+=0.5
                self.batch_lattice_ax.imshow(plot_package['current_T_lattice_copy'], cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation='nearest')

        # 更新右侧M(T)曲线
        if self.plot_M_var.get() and self.batch_M_vs_T_ax:
            self.batch_M_vs_T_ax.clear()
            if plot_package['batch_T_history'] and plot_package['batch_M_history']:
                self.batch_M_vs_T_ax.plot(plot_package['batch_T_history'], plot_package['batch_M_history'], marker='o', linestyle='-', ms=3, color='blue')
            self.batch_M_vs_T_ax.set_xlabel("Temperature T", fontsize=6); self.batch_M_vs_T_ax.set_ylabel("Final M", fontsize=6)
            self.batch_M_vs_T_ax.tick_params(axis='both', which='major', labelsize=6); self.batch_M_vs_T_ax.grid(True, ls=':', alpha=0.7)
            if plot_package['batch_T_history']: # 反转X轴，高温在左，低温在右
                self.batch_M_vs_T_ax.set_xlim(max(plot_package['batch_T_history'])+0.1, min(plot_package['batch_T_history'])-0.1)

        # 更新右侧E(T)曲线
        if self.plot_E_var.get() and self.batch_E_vs_T_ax:
            self.batch_E_vs_T_ax.clear()
            if plot_package['batch_T_history'] and plot_package['batch_E_history']:
                self.batch_E_vs_T_ax.plot(plot_package['batch_T_history'], plot_package['batch_E_history'], marker='o', linestyle='-', ms=3, color='red')
            self.batch_E_vs_T_ax.set_xlabel("Temperature T", fontsize=6); self.batch_E_vs_T_ax.set_ylabel("Final E", fontsize=6)
            self.batch_E_vs_T_ax.tick_params(axis='both', which='major', labelsize=6); self.batch_E_vs_T_ax.grid(True, ls=':', alpha=0.7)
            if plot_package['batch_T_history']:
                self.batch_E_vs_T_ax.set_xlim(max(plot_package['batch_T_history'])+0.1, min(plot_package['batch_T_history'])-0.1)

        self.fig.suptitle(suptitle_str, fontsize=12)
       
        self.canvas.draw_idle()

    def _save_snapshot(self, sim_id, temp_val, sweep_count, directory,model_config_at_start):
        if not self.root.winfo_exists(): return

        j_val = model_config_at_start.get('J_value',0.0); h_val = model_config_at_start.get('H',0.0)
        ns_val = model_config_at_start.get('num_states_for_ising_potts',2); rand_j = model_config_at_start.get('use_random_J',False)
        potts = model_config_at_start.get('use_potts_interaction',False); xy = model_config_at_start.get('use_xy_model',False)
        j_max = model_config_at_start.get('random_J_max', j_val)

        j_str_part = f"RandJpm{abs(j_max):.1f}" if rand_j else f"J{j_val:.1f}"
        model_type_part = "XY" if xy else ("Potts" if potts else "Ising")
        ns_part = f"Ns{ns_val}" if not xy else "XY"
        h_part = f"H{h_val:.2f}"
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        base_filename = f"snapshot_{model_type_part}_{j_str_part}_{ns_part}_{h_part}_T{temp_val:.3f}_sweeps{sweep_count}_{date_str}"
        
        try:
            filename = f"{base_filename}.png"
            filepath = os.path.join(directory, filename)
            self.fig.savefig(filepath)
            print(f"快照已保存到: {filepath}")
        except Exception as e:
            print(f"保存快照失败: {e}")
            messagebox.showerror("错误", f"保存快照失败: {e}")

    def _save_batch_data_to_file(self, sim_id, model_config_at_start):
        if not self.batch_temperatures_scanned:
            print("批量数据为空，不保存文件。")
            return

        data_dir = "./logs_data" # 数据保存的目录
        if not os.path.exists(data_dir):
            try: os.makedirs(data_dir)
            except OSError as e: print(f"创建数据目录 '{data_dir}' 失败: {e}"); messagebox.showerror("错误", f"创建数据目录失败: {e}"); return

        j_val = model_config_at_start.get('J_value',0.0); h_val = model_config_at_start.get('H',0.0)
        ns_val = model_config_at_start.get('num_states_for_ising_potts',2); rand_j = model_config_at_start.get('use_random_J',False)
        potts = model_config_at_start.get('use_potts_interaction',False); xy = model_config_at_start.get('use_xy_model',False)
        j_max = model_config_at_start.get('random_J_max', j_val)

        j_str_part = f"RandJpm{abs(j_max):.1f}" if rand_j else f"J{j_val:.1f}"
        model_type_part = "XY" if xy else ("Potts" if potts else "Ising")
        ns_part = f"Ns{ns_val}" if not xy else "XY"
        h_part = f"H{h_val:.2f}"
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        base_filename = f"batch_data_T{self.batch_temperatures_scanned[-1]:.2f}-{self.batch_temperatures_scanned[0]:.2f}_{model_type_part}_{j_str_part}_{ns_part}_{h_part}_{date_str}"
        safe_filename = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in base_filename) + ".csv"
        filepath = os.path.join(data_dir, safe_filename)

        try:
            with open(filepath, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["Temperature", "Magnetization_Final", "Energy_Final"])
                
                data_tuples = []
                for i in range(len(self.batch_temperatures_scanned)):
                    T_val = self.batch_temperatures_scanned[i]
                    M_val = self.batch_final_M_values[i] if i < len(self.batch_final_M_values) else float('nan')
                    E_val = self.batch_final_E_values[i] if i < len(self.batch_final_E_values) else float('nan')
                    data_tuples.append((T_val, M_val, E_val))
                
                data_tuples.sort(key=lambda x: x[0]) # 按温度升序排序
                
                for T_val_sorted, M_val_sorted, E_val_sorted in data_tuples:
                    writer.writerow([f"{T_val_sorted:.4f}", f"{M_val_sorted:.6f}", f"{E_val_sorted:.6f}"])
            print(f"批量实验数据已保存到: {filepath}")
        except Exception as e:
            print(f"保存批量数据文件失败: {e}")
            messagebox.showerror("错误", f"保存批量数据文件失败: {e}")
            
    def _finish_batch_mode(self, sim_id):
        if sim_id == self.current_simulation_id:
            if self.batch_save_data_var.get() and self.batch_temperatures_scanned:
                self._save_batch_data_to_file(sim_id, self.last_batch_model_config)
            self.is_batch_mode_active = False
            self._set_controls_state("normal")
            self.pause_button.config(text="暂停", state="disabled")
            self.reset_button.config(text="重置当前模拟", state="disabled")
            self.start_batch_button.config(text="开始批量实验", state="normal")
            #messagebox.showinfo("完成", f"批量实验 [任务 {sim_id}] 已完成")
            print(f"----- 批量实验 [任务 {sim_id}] 已完成 ! OwO-----")

    ### ----- 通用控制函数 -----

    def _toggle_pause_simulation(self):
        if self.pause_event.is_set(): self.pause_event.clear(); self.pause_button.config(text="继续"); print("模拟已暂停")
        else: self.pause_event.set(); self.pause_button.config(text="暂停"); print("模拟已继续")
    
    def _reset_current_simulation_action(self):
        if self.is_batch_mode_active:
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.stop_event.set()
                self.pause_event.set()
            self.is_batch_mode_active = False
            self._set_controls_state("normal")
            self.pause_button.config(text="暂停", state="disabled")
            self.reset_button.config(text="重置当前模拟", state="disabled")
            self.start_batch_button.config(text="开始批量实验", state="normal")
            self.fig.clear()
            self.canvas.draw_idle()
            print("批量实验已重置/停止。")
        else:
            self._start_or_reset_simulation(is_reset=True)

    def _on_closing(self): 
        if messagebox.askokcancel("退出", "您确定要退出吗？模拟将会停止"):
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.stop_event.set(); self.pause_event.set() 
                self.simulation_thread.join(timeout=0.5)
            self.root.destroy()