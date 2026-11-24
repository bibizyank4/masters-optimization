"""Графічна оболонка для локального дослідження методів."""
import json
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from optimizer.experiments import run_suite
from optimizer.research import sensitivity, matrix_comparison, compare_environments
from optimizer.final_analysis import final_analysis
from optimizer.starts import parse_extra_starts, validate_extra_starts
from optimizer.methods import Config, METHODS
from optimizer.problems import problem_suite
from optimizer.results import Dataset, METHOD_NAMES, PROBLEM_NAMES, STATUS_NAMES, TYPE_NAMES

ROOT_DIR = Path(__file__).resolve().parent
BG = '#f3f6fb'
WHITE = '#ffffff'
INK = '#15243b'
MUTED = '#596b83'
ACCENT = '#245bd7'
OK = '#147d64'
WARN = '#ad5d00'
BAD = '#b64343'
COLORS = ['#245bd7', '#e08722', '#168d79', '#c04c65', '#7f62bc', '#637b96', '#aa7415', '#bb3dab', '#496b29']


def fmt(value, digits=4):
    try:
        number = float(value)
        if not np.isfinite(number):
            return '—'
        return f'{number:.{digits}g}'
    except (TypeError, ValueError):
        return '—'


def number_or_none(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


class ResearchApp(tk.Tk):
    def __init__(self, open_path=None):
        super().__init__()
        self.title('Оптимум — порівняння методів мінімізації')
        width = min(1400, max(950, self.winfo_screenwidth() - 60))
        height = min(820, max(640, self.winfo_screenheight() - 120))
        self.geometry(f'{width}x{height}+20+15')
        self.minsize(950, 640)
        self.configure(bg=BG)
        self.dataset = None
        self.rows_by_tree = {}
        self.current_id = None
        self.events = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()
        self._style()
        self._variables()
        self._layout()
        self.protocol('WM_DELETE_WINDOW', self.close_window)
        self.after(100, self._poll)
        if open_path is not None:
            self.after(120, lambda: self.load_directory(open_path))
        elif (ROOT_DIR / 'example_results' / 'summary.csv').exists():
            self.after(120, lambda: self.load_directory(ROOT_DIR / 'example_results'))

    def _style(self):
        s = ttk.Style(self)
        if 'clam' in s.theme_names():
            s.theme_use('clam')
        s.configure('.', font=('Segoe UI', 10), foreground=INK, background=BG)
        s.configure('Card.TFrame', background=WHITE)
        s.configure('Card.TLabel', background=WHITE, foreground=INK)
        s.configure('Muted.TLabel', background=WHITE, foreground=MUTED, font=('Segoe UI', 9))
        s.configure('Title.TLabel', background=BG, foreground=INK, font=('Segoe UI', 23, 'bold'))
        s.configure('Subtitle.TLabel', background=BG, foreground=MUTED, font=('Segoe UI', 10))
        s.configure('Heading.TLabel', background=WHITE, foreground=INK, font=('Segoe UI', 12, 'bold'))
        s.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=(14, 9),
                    background=ACCENT, foreground='white', borderwidth=0)
        s.map('Accent.TButton', background=[('disabled', '#a8b9d7'), ('active', '#1b4fbe')],
              foreground=[('disabled', '#f4f6fa')])
        s.configure('TButton', padding=(9, 7))
        s.configure('TNotebook', background=BG, borderwidth=0)
        s.configure('TNotebook.Tab', padding=(17, 10), font=('Segoe UI', 10, 'bold'))
        s.map('TNotebook.Tab', background=[('selected', WHITE), ('!selected', BG)],
              foreground=[('selected', ACCENT), ('!selected', MUTED)])
        s.configure('Treeview', background=WHITE, fieldbackground=WHITE, rowheight=30,
                    foreground=INK, borderwidth=0, font=('Segoe UI', 9))
        s.configure('Treeview.Heading', background='#eaf0f9', foreground='#334860',
                    font=('Segoe UI', 9, 'bold'), padding=(6, 8))
        s.map('Treeview', background=[('selected', '#dce8ff')], foreground=[('selected', INK)])
        s.configure('Horizontal.TProgressbar', troughcolor='#e4eaf3', background=ACCENT,
                    borderwidth=0, thickness=8)

    def _variables(self):
        self.problem_options = {'Усі задачі': 'all'}
        for p in problem_suite():
            label = f'{PROBLEM_NAMES[p.name]} · n={p.dimension}'
            self.problem_options[label] = f'{p.name}:{p.dimension}'
        self.method_options = {'Усі методи': 'all', 'Три трикрокові методи': 'three_step_family',
                               **{METHOD_NAMES[m]: m for m in METHODS}}
        self.problem_var = tk.StringVar(value='Усі задачі')
        self.method_var = tk.StringVar(value='Усі методи')
        self.start_var = tk.StringVar(value='Усі')
        self.repeats_var = tk.StringVar(value='3')
        self.warmups_var = tk.StringVar(value='2')
        self.tol_var = tk.StringVar(value='1e-8')
        self.max_var = tk.StringVar(value='500')
        self.b0_var = tk.StringVar(value='Гессіан у старті')
        self.bfgs_var = tk.StringVar(value='Пропуск')
        self.bfgs_tol_var = tk.StringVar(value='1e-10')
        self.restart_var = tk.StringVar(value='0')
        self.search_var = tk.StringVar(value='Просте спадання')
        self.experiment_var = tk.StringVar(value='Звичайна серія')
        self.scales_var = tk.StringVar(value='1e-15, 1e-12, 1e-8')
        self.samples_var = tk.StringVar(value='3')
        self.seed_var = tk.StringVar(value='42')
        self.extra_starts_var = tk.StringVar(value='')
        self.plot_var = tk.StringVar(value='Норма градієнта')
        self.filter_var = tk.StringVar()
        self.dataset_var = tk.StringVar(value='Результати ще не відкриті')
        self.footer_var = tk.StringVar(value='Оберіть параметри та запустіть експеримент або відкрийте результати.')
        self.detail_var = tk.StringVar(value='Оберіть рядок таблиці для детальної інформації.')
        self.graph_title_var = tk.StringVar(value='Оберіть рядок у таблиці результатів')
        self.progress_var = tk.DoubleVar(value=0)

    def _layout(self):
        header = ttk.Frame(self, padding=(24, 18, 24, 12))
        header.pack(fill='x')
        ttk.Label(header, text='Оптимум', style='Title.TLabel').pack(side='left')
        ttk.Label(header, text='  /  дослідження багатокрокових методів', style='Subtitle.TLabel').pack(side='left', anchor='s', pady=(0, 6))
        ttk.Label(header, text='МАГІСТЕРСЬКА  ·  v3.4', style='Subtitle.TLabel').pack(side='right', anchor='s', pady=(0, 7))

        body = ttk.Frame(self, padding=(20, 0, 20, 8))
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        side_shell = ttk.Frame(body, style='Card.TFrame', width=300)
        side_shell.grid(row=0, column=0, sticky='ns', padx=(0, 12))
        side_shell.grid_propagate(False)
        side_shell.columnconfigure(0, weight=1)
        side_shell.rowconfigure(0, weight=1)
        side_canvas = tk.Canvas(side_shell, bg=WHITE, borderwidth=0, highlightthickness=0, width=275)
        side_canvas.grid(row=0, column=0, sticky='nsew')
        side_scroll = ttk.Scrollbar(side_shell, orient='vertical', command=side_canvas.yview)
        side_scroll.grid(row=0, column=1, sticky='ns')
        side_canvas.configure(yscrollcommand=side_scroll.set)
        sidebar = ttk.Frame(side_canvas, style='Card.TFrame', padding=17)
        side_window = side_canvas.create_window((0, 0), window=sidebar, anchor='nw')
        sidebar.bind('<Configure>', lambda event: side_canvas.configure(scrollregion=side_canvas.bbox('all')))
        side_canvas.bind('<Configure>', lambda event: side_canvas.itemconfigure(side_window, width=event.width))
        self._controls(sidebar)
        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        info = ttk.Frame(main, style='Card.TFrame', padding=(14, 10))
        info.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(info, text='СЕРІЯ', style='Muted.TLabel').pack(side='left', padx=(0, 10))
        ttk.Label(info, textvariable=self.dataset_var, style='Card.TLabel').pack(side='left', fill='x', expand=True)
        ttk.Button(info, text='Відкрити результати…', command=self.choose_directory).pack(side='right')

        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=1, column=0, sticky='nsew')
        table_page = ttk.Frame(self.tabs, style='Card.TFrame', padding=14)
        graph_page = ttk.Frame(self.tabs, style='Card.TFrame', padding=14)
        self.tabs.add(table_page, text='  Таблиця результатів  ')
        self.tabs.add(graph_page, text='  Графіки  ')
        self._table(table_page)
        self._graphs(graph_page)
        footer = ttk.Frame(self, padding=(24, 5, 24, 14))
        footer.pack(fill='x')
        ttk.Progressbar(footer, variable=self.progress_var, maximum=100, length=170).pack(side='right', padx=(14, 0))
        ttk.Label(footer, textvariable=self.footer_var, style='Subtitle.TLabel').pack(side='left', fill='x', expand=True)

    def _controls(self, frame):
        ttk.Label(frame, text='Новий експеримент', style='Heading.TLabel').pack(anchor='w')
        ttk.Label(frame, text='Методи, задачі та умови запуску', style='Muted.TLabel').pack(anchor='w', pady=(2, 13))
        def choice(label, variable, values):
            ttk.Label(frame, text=label, style='Card.TLabel').pack(anchor='w', pady=(6, 3))
            box = ttk.Combobox(frame, textvariable=variable, values=values, state='readonly')
            box.pack(fill='x')
            return box
        choice('Тип експерименту', self.experiment_var,
               ['Звичайна серія', 'Комплексний фінальний аналіз', 'Порівняння B₀ і BFGS', 'Чутливість до старту', 'Порівняти збережені серії'])
        problem_box = choice('Задача', self.problem_var, list(self.problem_options))
        problem_box.bind('<<ComboboxSelected>>', lambda e: self._update_starts())
        self.start_box = choice('Початкова точка', self.start_var, ['Усі'])
        ttk.Label(frame, text='Додаткові точки через ;', style='Card.TLabel').pack(anchor='w', pady=(6, 3))
        ttk.Entry(frame, textvariable=self.extra_starts_var).pack(fill='x')
        ttk.Label(frame, text='Напр. для n=20: (-1,1)*10; (0,0)*10', style='Muted.TLabel', wraplength=240).pack(anchor='w')
        choice('Метод', self.method_var, list(self.method_options))
        row = ttk.Frame(frame, style='Card.TFrame')
        row.pack(fill='x', pady=(8, 0))
        for label, var in [('Точність ‖∇f‖', self.tol_var), ('Ітерацій', self.max_var)]:
            col = ttk.Frame(row, style='Card.TFrame')
            col.pack(side='left', fill='x', expand=True, padx=(0, 8))
            ttk.Label(col, text=label, style='Card.TLabel').pack(anchor='w')
            ttk.Entry(col, textvariable=var, width=10).pack(fill='x', pady=(3, 0))
        row2 = ttk.Frame(frame, style='Card.TFrame')
        row2.pack(fill='x', pady=(9, 0))
        ttk.Label(row2, text='Повторень', style='Card.TLabel').pack(side='left')
        ttk.Spinbox(row2, from_=1, to=100, textvariable=self.repeats_var, width=6).pack(side='right')
        row3 = ttk.Frame(frame, style='Card.TFrame')
        row3.pack(fill='x', pady=(9, 0))
        ttk.Label(row3, text='Прогрівів', style='Card.TLabel').pack(side='left')
        ttk.Spinbox(row3, from_=0, to=100, textvariable=self.warmups_var, width=6).pack(side='right')
        choice('Початкова матриця B₀', self.b0_var, ['Гессіан у старті', 'Одинична'])
        choice('BFGS: умова кривизни', self.bfgs_var, ['Пропуск', 'Демпфування Пауелла'])
        ttk.Label(frame, text='BFGS: відносний поріг кривизни', style='Card.TLabel').pack(anchor='w', pady=(6, 3))
        ttk.Entry(frame, textvariable=self.bfgs_tol_var).pack(fill='x')
        ttk.Label(frame, text='Бройден: оновлення гессіана (0 = вимкнено)', style='Card.TLabel', wraplength=240).pack(anchor='w', pady=(6, 3))
        ttk.Entry(frame, textvariable=self.restart_var).pack(fill='x')
        choice('Крок коректора', self.search_var, ['Просте спадання', 'Арміхо'])
        ttk.Label(frame, text='Чутливість: масштаби через кому', style='Card.TLabel').pack(anchor='w', pady=(6, 3))
        ttk.Entry(frame, textvariable=self.scales_var).pack(fill='x')
        row4 = ttk.Frame(frame, style='Card.TFrame')
        row4.pack(fill='x', pady=(8, 0))
        for label, var in [('Збурень', self.samples_var), ('Seed через кому', self.seed_var)]:
            col = ttk.Frame(row4, style='Card.TFrame')
            col.pack(side='left', fill='x', expand=True, padx=(0, 8))
            ttk.Label(col, text=label, style='Card.TLabel').pack(anchor='w')
            ttk.Entry(col, textvariable=var, width=10).pack(fill='x')
        ttk.Label(frame, text='Для трикрокового методу використовується крок Арміхо.',
                  style='Muted.TLabel', wraplength=240, justify='left').pack(anchor='w', pady=(10, 8))
        ttk.Label(frame, text='B₀/BFGS: 4 серії для обраних задач і методів. Чутливість: по одному запуску на старт; прогріви і повтори не використовуються.',
                  style='Muted.TLabel', wraplength=240, justify='left').pack(anchor='w', pady=(0, 8))
        ttk.Label(frame, text='Фінальний аналіз: усі вбудовані задачі, 3 точності, 20 повторів, 2 прогріви та чутливість; може тривати кілька годин.',
                  style='Muted.TLabel', wraplength=240, justify='left').pack(anchor='w', pady=(0, 8))
        self.run_button = ttk.Button(frame, text='▶  Запустити експеримент', style='Accent.TButton', command=self.start_run)
        self.run_button.pack(fill='x', pady=(5, 5))
        self.cancel_button = ttk.Button(frame, text='Зупинити після поточного запуску', command=self.cancel_run, state='disabled')
        self.cancel_button.pack(fill='x')
        ttk.Label(frame, text='Результати кожної серії зберігаються в окремій папці runs.',
                  style='Muted.TLabel', wraplength=240, justify='left').pack(anchor='w', pady=(12, 0))

    def _update_starts(self):
        key = self.problem_options[self.problem_var.get()]
        if key == 'all':
            values = ['Усі']
        else:
            name, n = key.rsplit(':', 1)
            p = next(p for p in problem_suite() if p.name == name and p.dimension == int(n))
            values = ['Усі'] + [f'{i}: {tuple(x)}' for i, x in enumerate(p.starts)]
        self.start_box.configure(values=values)
        self.start_var.set('Усі')

    def _table(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        top = ttk.Frame(frame, style='Card.TFrame')
        top.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        ttk.Label(top, text='Порівняння запусків', style='Heading.TLabel').pack(side='left')
        ttk.Label(top, text='Пошук:', style='Muted.TLabel').pack(side='left', padx=(20, 5))
        search = ttk.Entry(top, textvariable=self.filter_var, width=28)
        search.pack(side='left')
        self.filter_var.trace_add('write', lambda *_: self.populate_table())
        ttk.Button(top, text='Показати графік →', command=lambda: self.tabs.select(1)).pack(side='right')
        columns = ('problem', 'start', 'method', 'status', 'iterations', 'f', 'grad', 'seconds')
        self.tree = ttk.Treeview(frame, columns=columns, show='headings', selectmode='browse')
        definitions = {'problem': ('Задача', 150), 'start': ('Старт', 52), 'method': ('Метод', 165),
                       'status': ('Результат', 190), 'iterations': ('Ітер.', 65), 'f': ('f(x)', 90),
                       'grad': ('‖∇f‖', 95), 'seconds': ('Час, с', 80)}
        for key, (label, width) in definitions.items():
            self.tree.heading(key, text=label, command=lambda k=key: self.sort_table(k))
            self.tree.column(key, width=width, minwidth=45, anchor='w' if key in ('problem','method','status') else 'e')
        self.tree.tag_configure('warning', foreground=WARN)
        self.tree.tag_configure('failure', foreground=BAD)
        self.tree.grid(row=1, column=0, sticky='nsew')
        ybar = ttk.Scrollbar(frame, orient='vertical', command=self.tree.yview)
        ybar.grid(row=1, column=1, sticky='ns')
        xbar = ttk.Scrollbar(frame, orient='horizontal', command=self.tree.xview)
        xbar.grid(row=2, column=0, sticky='ew')
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        details = ttk.Frame(frame, style='Card.TFrame', padding=(2, 12, 2, 0))
        details.grid(row=3, column=0, columnspan=2, sticky='ew')
        ttk.Label(details, text='Обраний запуск', style='Heading.TLabel').pack(anchor='w')
        ttk.Label(details, textvariable=self.detail_var, style='Card.TLabel', justify='left',
                  wraplength=1000).pack(anchor='w', pady=(5, 0))
        ttk.Label(details, text='Стаціонарна точка не обов’язково є мінімумом. Порядок збіжності — лише чисельна оцінка.',
                  style='Muted.TLabel', wraplength=1000).pack(anchor='w', pady=(7, 0))

    def _graphs(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        top = ttk.Frame(frame, style='Card.TFrame')
        top.grid(row=0, column=0, sticky='ew')
        ttk.Label(top, textvariable=self.graph_title_var, style='Heading.TLabel').pack(side='left', fill='x', expand=True)
        box = ttk.Combobox(top, textvariable=self.plot_var, state='readonly', width=25,
                           values=['Норма градієнта', 'Значення функції', 'Траєкторії · 2D',
                                   'Кількість ітерацій', 'Середній час'])
        box.pack(side='right')
        box.bind('<<ComboboxSelected>>', lambda e: self.draw_plot())
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=WHITE, layout='constrained')
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        bottom = ttk.Frame(frame, style='Card.TFrame')
        bottom.grid(row=2, column=0, sticky='ew')
        toolbar = NavigationToolbar2Tk(self.canvas, bottom, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side='left')
        ttk.Label(bottom, text='Обрана задача та старт; усі доступні методи. Значок дискети — зберегти графік.',
                  style='Muted.TLabel').pack(side='left', padx=8)
        self._empty_plot('Виберіть запуск у таблиці')

    def _empty_plot(self, message):
        self.axes.clear()
        self.axes.axis('off')
        self.axes.text(.5, .5, message, transform=self.axes.transAxes,
                       ha='center', va='center', color=MUTED, fontsize=12)
        self.canvas.draw_idle()

    def choose_directory(self):
        path = filedialog.askdirectory(title='Оберіть папку з файлами summary.csv, history.csv та metadata.json')
        if path:
            if (Path(path) / 'REPORT.md').is_file() and not (Path(path) / 'summary.csv').exists():
                self.show_report(path)
            else:
                self.load_directory(path)

    def show_report(self, path):
        from tkinter.scrolledtext import ScrolledText
        report = Path(path) / 'REPORT.md'
        try:
            contents = report.read_text(encoding='utf-8')
        except OSError as exc:
            messagebox.showerror('Не вдалося відкрити звіт', str(exc), parent=self)
            return
        window = tk.Toplevel(self)
        window.title('Звіт експерименту')
        window.geometry('1000x650')
        ttk.Label(window, text=str(report), wraplength=950).pack(fill='x', padx=10, pady=8)
        text = ScrolledText(window, wrap='none', font=('Consolas', 10))
        text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        text.insert('1.0', contents)
        text.configure(state='disabled')

    def load_directory(self, path):
        try:
            data = Dataset(path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            messagebox.showerror('Не вдалося відкрити результати', str(exc), parent=self)
            return
        self.dataset = data
        self.current_id = None
        marker = 'СТАРА ВЕРСІЯ: градієнтний коректор' if data.legacy else 'Коректор з H⁻¹ / B⁻¹ · v' + str(data.metadata.get('program_version', '?'))
        if Path(path).resolve() == (ROOT_DIR / 'example_results').resolve():
            marker = 'Демонстраційна серія · ' + marker
        if data.metadata.get('cancelled'):
            marker += ' · НЕПОВНА СЕРІЯ'
        repeats = data.metadata.get('repeats', '—')
        warmups = data.metadata.get('warmups', 'не вказано')
        self.dataset_var.set(f'{marker}  |  {len(data.rows)} запусків  |  повторів: {repeats}  |  прогрівів: {warmups}  |  {Path(path).name}')
        self.footer_var.set('Завантажено: ' + str(Path(path).resolve()))
        self.progress_var.set(0)
        self.populate_table()
        self._empty_plot('Виберіть запуск у таблиці')
        self.graph_title_var.set('Оберіть рядок у таблиці результатів')
        self.detail_var.set('Оберіть рядок таблиці для детальної інформації.')

    def populate_table(self):
        if not hasattr(self, 'tree'):
            return
        self.tree.delete(*self.tree.get_children())
        self.rows_by_tree.clear()
        if self.dataset is None:
            return
        query = self.filter_var.get().strip().casefold()
        for i, row in enumerate(self.dataset.rows):
            state = row['status']
            classification = row['stationary_type']
            status = STATUS_NAMES.get(state, state)
            if state == 'stationary' and classification == 'negative_hessian_eigenvalue':
                status = 'Стаціонарна · не мінімум'
            elif state == 'stationary' and classification == 'inconclusive_hessian':
                status = 'Стаціонарна · перевірити'
            values = (self.dataset.problem_label(row), row['start_index'], self.dataset.method_label(row), status,
                      row['iterations'], fmt(row['f']), fmt(row['grad_norm']), fmt(row.get('time_mean_seconds')))
            if query and query not in ' '.join(map(str, values)).casefold():
                continue
            tag = 'failure' if state != 'stationary' or classification == 'negative_hessian_eigenvalue' else (
                  'warning' if classification == 'inconclusive_hessian' else '')
            item = self.tree.insert('', 'end', iid=f'row-{i}', values=values, tags=(tag,))
            self.rows_by_tree[item] = row['case_id']

    def sort_table(self, key):
        items = list(self.tree.get_children())
        numeric = key in ('start', 'iterations', 'f', 'grad', 'seconds')
        idx = self.tree['columns'].index(key)
        def val(item):
            v = self.tree.item(item, 'values')[idx]
            if numeric:
                n = number_or_none(v)
                return float('inf') if n is None else n
            return v.casefold()
        reverse = getattr(self, '_sort_state', (None, True)) == (key, False)
        for position, item in enumerate(sorted(items, key=val, reverse=reverse)):
            self.tree.move(item, '', position)
        self._sort_state = (key, reverse)

    def on_select(self, event=None):
        selection = self.tree.selection()
        if not selection or not self.dataset:
            return
        self.current_id = self.rows_by_tree.get(selection[0])
        if not self.current_id:
            return
        row = self.dataset.by_id[self.current_id]
        info = [f"{self.dataset.problem_label(row)}  ·  старт {row['start_index']}  ·  {self.dataset.method_label(row)}",
                f"Статус: {STATUS_NAMES.get(row['status'], row['status'])};  {TYPE_NAMES.get(row['stationary_type'], row['stationary_type'])}",
                f"x₀ = {row.get('start', '—')}    x = {row['x']}",
                f"‖∇f‖ = {fmt(row['grad_norm'], 7)}   f(x) = {fmt(row['f'], 7)}   відстань до відомого мінімуму = {fmt(row.get('distance_to_nearest_minimum'), 7)}",
                f"Обчислень: f = {row.get('calls_f', '—')}, ∇f = {row.get('calls_grad', '—')}, H = {row.get('calls_hess', '—')}, систем = {row.get('calls_linear_solves', '—')}",
                f"BFGS: звичайних оновлень = {row.get('calls_bfgs_applied', '—')}, пропущено = {row.get('calls_bfgs_skipped', '—')}, демпфовано = {row.get('calls_bfgs_damped', '—')}",
                f"Бройден: перезапусків з гессіаном = {row.get('calls_broyden_restarts', '—')}",
                f"Час: середній {fmt(row.get('time_mean_seconds'), 6)} с, медіана {fmt(row.get('time_median_seconds'), 6)} с;  p ≈ {fmt(row.get('observed_order'))} (оцінок: {row.get('order_estimates', '—')})"]
        if row.get('message'):
            info.append('Пояснення: ' + row['message'])
        self.detail_var.set('\n'.join(info))
        self.graph_title_var.set(f"{self.dataset.problem_label(row)}  ·  старт {row['start_index']}")
        self.draw_plot()

    def draw_plot(self):
        if not self.dataset or not self.current_id:
            self._empty_plot('Виберіть запуск у таблиці')
            return
        selected = self.dataset.by_id[self.current_id]
        rows = self.dataset.comparison(selected)
        def plot_label(row):
            label = self.dataset.method_label(row)
            if row['status'] != 'stationary':
                return label + ' · зупинка'
            if row['stationary_type'] == 'negative_hessian_eigenvalue':
                return label + ' · не мінімум'
            if row['stationary_type'] == 'inconclusive_hessian':
                return label + ' · перевірити'
            return label
        ax = self.axes
        ax.clear()
        ax.set_facecolor('#fbfcff')
        kind = self.plot_var.get()
        ax.grid(True, color='#dfe6f0', linewidth=.7, alpha=.8)
        if kind in ('Норма градієнта', 'Значення функції'):
            label = 'gradient_norm' if kind == 'Норма градієнта' else 'f_value'
            # Поліноми в наборі є невід’ємними. Нулі для логарифмічної шкали лише відображаємо як 1e-16.
            for i, row in enumerate(rows):
                hist = self.dataset.histories.get(row['case_id'], [])
                if hist:
                    ax.semilogy([h['k'] for h in hist], [max(h[label], 1e-16) for h in hist],
                                color=COLORS[i % len(COLORS)], linewidth=1.6,
                                label=plot_label(row))
            ax.set_xlabel('Ітерація')
            ax.set_ylabel('‖∇f(x)‖' if label == 'gradient_norm' else 'f(x)')
            ax.set_title('Нижня межа відображення: 10⁻¹⁶', fontsize=9, color=MUTED)
            ax.legend(loc='best', fontsize=8)
        elif kind == 'Траєкторії · 2D':
            if int(selected['dimension']) != 2:
                self._empty_plot('Траєкторії доступні для задач із двома змінними')
                return
            problem = self.dataset.problems.get((selected['problem'], 2))
            if problem is None:
                self._empty_plot('Немає функції для побудови ліній рівня')
                return
            initial = np.array(json.loads(selected['start']), float)
            anchor = np.array(min(problem.minima, key=lambda x: np.linalg.norm(initial-np.asarray(x))), float)
            span = max(2., float(np.max(np.abs(initial-anchor)))+1.)
            low = np.minimum(np.minimum(initial, anchor)-span*.35, initial-1)
            high = np.maximum(np.maximum(initial, anchor)+span*.35, initial+1)
            xx, yy = np.meshgrid(np.linspace(low[0], high[0], 140), np.linspace(low[1], high[1], 140))
            z = problem.expression([xx, yy])
            positive = z[z > 0]
            if len(positive):
                levels = np.geomspace(max(float(np.min(positive)), 1e-5), max(float(np.max(z)), 1e-4), 23)
                ax.contour(xx, yy, z, levels=levels, colors='#c3ccda', linewidths=.6)
            for i, row in enumerate(rows):
                hist = self.dataset.histories.get(row['case_id'], [])
                if hist:
                    points = np.asarray([h['point'] for h in hist], dtype=float)
                    ax.plot(points[:, 0], points[:, 1], '.-', linewidth=1.3, markersize=3,
                            color=COLORS[i % len(COLORS)], label=plot_label(row))
            ax.scatter(initial[0], initial[1], marker='x', s=65, color=INK, zorder=5, label='Старт')
            for x in problem.minima:
                if low[0] <= x[0] <= high[0] and low[1] <= x[1] <= high[1]:
                    ax.scatter(x[0], x[1], marker='*', s=80, color=BAD, zorder=5)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='x₁', ylabel='x₂')
            ax.set_title('Точки поза вікном обрізані', fontsize=9, color=MUTED)
            ax.legend(loc='best', fontsize=7)
        else:
            metric = 'iterations' if kind == 'Кількість ітерацій' else 'time_mean_seconds'
            values = [float(r.get(metric, 0) or 0) for r in rows]
            labels = [self.dataset.method_label(r) for r in rows]
            bars = ax.barh(labels[::-1], values[::-1], color=[COLORS[i % len(COLORS)] for i in range(len(rows))][::-1])
            # Статус видно без помилкового трактування швидкої невдачі як перемоги.
            for bar, row in zip(bars, rows[::-1]):
                if row['status'] != 'stationary' or row['stationary_type'] != 'positive_definite_hessian':
                    bar.set_hatch('///')
                    bar.set_alpha(.6)
            ax.set_xlabel('Ітерацій' if metric == 'iterations' else 'Секунд, середнє')
            ax.set_title('Штрихування — зупинка без підтвердженого мінімуму', fontsize=9, color=MUTED)
            if metric == 'iterations':
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.canvas.draw_idle()

    def _read_config(self):
        tol, maximum, repeats = float(self.tol_var.get()), int(self.max_var.get()), int(self.repeats_var.get())
        warmups = int(self.warmups_var.get())
        if not 1 <= repeats <= 100:
            raise ValueError('Кількість повторень має бути від 1 до 100')
        if not 0 <= warmups <= 100:
            raise ValueError('Кількість прогрівів має бути від 0 до 100')
        cfg = Config(grad_tol=tol, max_iterations=maximum,
                     b0='hessian' if self.b0_var.get() == 'Гессіан у старті' else 'identity',
                     bfgs_strategy='skip' if self.bfgs_var.get() == 'Пропуск' else 'damp',
                     bfgs_min_curvature=float(self.bfgs_tol_var.get()),
                     broyden_restart_interval=int(self.restart_var.get()),
                     correction_search='monotone' if self.search_var.get() == 'Просте спадання' else 'armijo')
        cfg.validate()
        start = self.start_var.get()
        return cfg, repeats, warmups, self.problem_options[self.problem_var.get()], self.method_options[self.method_var.get()], (
               None if start == 'Усі' else int(start.split(':', 1)[0]))

    def start_run(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            kind = self.experiment_var.get()
            if kind == 'Комплексний фінальний аналіз':
                cfg, repeats, warmups, problem, method, start = Config(), 20, 2, 'all', 'all', None
            else:
                cfg, repeats, warmups, problem, method, start = self._read_config()
            sources = None
            extra_starts = parse_extra_starts(self.extra_starts_var.get()) if kind not in ('Порівняти збережені серії', 'Комплексний фінальний аналіз') else []
            if extra_starts:
                selected_suite = [p for p in problem_suite() if problem in (p.name, f'{p.name}:{p.dimension}')]
                extra_starts = validate_extra_starts(selected_suite, extra_starts)
            if kind == 'Чутливість до старту':
                scales = tuple(float(s.strip()) for s in self.scales_var.get().split(','))
                samples = int(self.samples_var.get())
                seeds = tuple(int(s.strip()) for s in self.seed_var.get().split(','))
                if not scales or len(set(scales)) != len(scales) or any(not np.isfinite(s) or s <= 0 for s in scales):
                    raise ValueError('Масштаби збурень мають бути різними додатними числами')
                if not 1 <= samples <= 100 or not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
                    raise ValueError('Кількість збурень: 1–100; seed: різні невід’ємні цілі')
            elif kind == 'Порівняти збережені серії':
                first = filedialog.askdirectory(title='Оберіть першу папку зі збереженою серією', parent=self)
                if not first:
                    return
                second = filedialog.askdirectory(title='Оберіть другу папку зі збереженою серією', parent=self)
                if not second:
                    return
                sources = (first, second)
            root = ROOT_DIR / 'runs'
            root.mkdir(parents=True, exist_ok=True)
            outdir = root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            outdir.mkdir(exist_ok=False)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Перевірте параметри', str(exc), parent=self)
            return
        self.stop_event.clear()
        self.run_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress_var.set(0)
        self.footer_var.set('Виконується експеримент…')
        def progress(done, total, row):
            self.events.put(('progress', done, total, row))
        def work():
            try:
                if kind == 'Звичайна серія':
                    result = run_suite(outdir, cfg, repeats, problem, method, start,
                                       progress=progress, cancel_check=self.stop_event.is_set, verbose=False,
                                       warmups=warmups, extra_starts=extra_starts)
                    self.events.put(('done', outdir, len(result), self.stop_event.is_set()))
                else:
                    if kind == 'Комплексний фінальний аналіз':
                        result = final_analysis(outdir, repeats=20, warmups=2,
                                                progress=progress, cancel_check=self.stop_event.is_set)
                    elif kind == 'Порівняння B₀ і BFGS':
                        result = matrix_comparison(outdir, cfg, repeats, warmups, problem, method, start,
                                                   progress, self.stop_event.is_set, extra_starts=extra_starts)
                    elif kind == 'Чутливість до старту':
                        result = sensitivity(outdir, cfg, problem, method, start, scales, samples,
                                             progress=progress, cancel_check=self.stop_event.is_set,
                                             seeds=seeds, extra_starts=extra_starts)
                    else:
                        result = compare_environments(outdir, *sources)
                    self.events.put(('done_report', outdir, result))
            except Exception:
                self.events.put(('error', traceback.format_exc()))
        self.worker = threading.Thread(target=work, daemon=True, name='experiments')
        self.worker.start()

    def cancel_run(self):
        self.stop_event.set()
        self.cancel_button.configure(state='disabled')
        self.footer_var.set('Зупиняємо після поточного повторення; завершені результати буде збережено…')

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'progress':
                    _, done, total, row = event
                    self.progress_var.set(100 * done / total)
                    self.footer_var.set(f"Обчислено {done} / {total}: {row['problem']} · {METHOD_NAMES.get(row['method'], row['method'])}")
                elif event[0] == 'done':
                    _, outdir, count, was_cancelled = event
                    self.run_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    if count:
                        self.load_directory(outdir)
                        self.footer_var.set(f"{'Зупинено' if was_cancelled else 'Готово'}: {count} запусків. Папка: {outdir}")
                        if not was_cancelled:
                            self.progress_var.set(100)
                    else:
                        self.footer_var.set('Експеримент зупинено до завершення першої серії повторень.')
                elif event[0] == 'done_report':
                    _, outdir, result = event
                    self.run_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    result_status = result.get('status', 'cancelled' if result.get('cancelled') else 'complete')
                    if result_status == 'complete':
                        self.progress_var.set(100)
                    label = {'complete': 'Готово: ', 'cancelled': 'Зупинено: ',
                             'failed': 'Помилка: ', 'validation_failed': 'Автотести не пройшли: '}.get(result_status, 'Стан: ')
                    self.footer_var.set(label + str(outdir))
                    self.show_report(outdir)
                elif event[0] == 'error':
                    self.run_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    self.footer_var.set('Не вдалося завершити експеримент.')
                    messagebox.showerror('Помилка обчислення', event[1], parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def close_window(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno('Обчислення триває',
                                       'Зупинити обчислення та закрити програму? Незбережені результати буде втрачено.',
                                       parent=self):
                return
            self.stop_event.set()
        self.destroy()


if __name__ == '__main__':
    ResearchApp().mainloop()
