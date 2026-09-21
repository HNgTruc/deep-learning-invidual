# Ghi chú học tập — Lab 01 & Lab 02

## 1. Lab 01 học gì

Môi trường: **Anaconda + Python + JupyterLab**

- **Bài 1 — Cài đặt & làm quen môi trường**
  - Cài Anaconda, thiết lập biến môi trường PATH
  - Tạo môi trường ảo riêng (`conda create -n ai_prj python=3.11`)
  - Cài Jupyter, NumPy, Matplotlib, SciPy, scikit-learn
  - Làm quen JupyterLab: tạo notebook, phân biệt Markdown cell / Code cell
  - Bài tập: sinh 50 hình tròn ngẫu nhiên (`ex2_scientific1`), vẽ đồ thị sin/cos (`ex4_scientific2`), chạy file `.py` từ notebook bằng `%run` (`calc.py` + `main_calc.ipynb`)

- **Bài 2 — Cơ bản về ngôn ngữ Python**
  - Nhập/xuất dữ liệu (`input`, `print`, f-string)
  - Biểu thức toán học, thư viện `math`
  - Cấu trúc rẽ nhánh `if/elif/else`
  - Cấu trúc lặp `for`, `while`
  - Xử lý ngoại lệ `try/except`
  - Định nghĩa hàm (`def`), tham số mặc định
  - Kiểu dữ liệu nâng cao: List, Tuple, Set, Dictionary

- **Bài 3 — Kỹ thuật lập trình Python (NumPy & Matplotlib)**
  - Tạo và thao tác mảng (`np.array`, indexing, slicing, broadcasting)
  - Thống kê trên mảng: max, min, trung bình, phương sai, độ lệch chuẩn
  - Ma trận: chuyển vị, nhân ma trận (`np.matmul`), giải hệ phương trình tuyến tính (`np.linalg.solve`)
  - Sinh số ngẫu nhiên theo phân bố đều / phân bố chuẩn, vẽ histogram
  - Vẽ đồ thị với `matplotlib.pyplot` (scatter, bar, line)
  - Ứng dụng: tính gần đúng số π bằng phương pháp Monte Carlo

## 2. Lab 02 học gì

Môi trường: **PyTorch** (thư viện Deep Learning)

- Tạo Tensor cơ bản, các Tensor thường dùng (`rand`, `randn`, `zeros`, `ones`, `eye`)
- Kiểm tra thuộc tính Tensor: `shape`, `size()`, `ndim`, `dtype`
- Indexing, slicing trên Tensor nhiều chiều
- Reshape, flatten (`reshape`, `ravel`)
- Phép toán phần tử và nhân ma trận (`*` vs `@`)
- Hàm thống kê trên Tensor: `mean`, `std`, `cumsum` theo từng chiều (`dim`)
- **Autograd** — cơ chế tính đạo hàm tự động (`requires_grad=True`, `.backward()`, `.grad`)
- **Gradient Descent** với optimizer `torch.optim.NAdam` để tìm hệ số đa thức, giải hệ phương trình

## 3. Cách trỏ vào môi trường `ai_prj`

Môi trường `ai_prj` là môi trường ảo (virtual environment) riêng, chứa Python + các thư viện cần dùng, tách biệt với môi trường `base` mặc định của Anaconda.

### Cách 1 — Dùng Command Prompt (cmd) đã init sẵn

```
conda activate ai_prj
```

Dấu nhắc lệnh sẽ đổi thành:

```
(ai_prj) D:\...>
```

### Cách 2 — Nếu đang ở PowerShell (báo lỗi "conda not recognized")

PowerShell thường chưa được "init" cho conda, dùng lệnh gọi thẳng file `activate.bat`:

```powershell
cmd /k "C:\Users\hntruc\anaconda3\Scripts\activate.bat ai_prj"
```

Lệnh này mở một phiên cmd con bên trong, tự kích hoạt luôn môi trường `ai_prj`.

### Cách 3 — Init conda cho cmd (chỉ cần làm 1 lần)

```
conda init cmd.exe
```

Sau đó đóng cửa sổ, mở cmd mới, `conda activate ai_prj` sẽ chạy bình thường từ đó về sau.

### Kiểm tra đã vào đúng môi trường chưa

```
python -c "import sys; print(sys.executable)"
```

Nếu đường dẫn in ra có chữ `envs\ai_prj` (ví dụ `...\anaconda3\envs\ai_prj\python.exe`) → đã đúng.

## 4. Cách chạy chương trình trên JupyterLab

### Bước 1 — Kích hoạt môi trường và mở JupyterLab

```
conda activate ai_prj
cd /d "duong-dan-toi-thu-muc-du-an"
jupyter lab
```

Trình duyệt sẽ tự mở giao diện JupyterLab tại `localhost:8888`.

### Bước 2 — Mở file `.ipynb` cần chạy

Nhấp đúp vào file trong khung File Browser bên trái.

### Bước 3 — Chạy từng ô (cell)

- **Shift + Enter** — chạy ô hiện tại, tự động chuyển sang ô kế tiếp
- **Ctrl + Enter** — chạy ô hiện tại, giữ nguyên vị trí
- Nút **▶ Run** trên thanh công cụ cũng chạy được ô đang chọn

### Bước 4 — Chạy toàn bộ notebook

Menu **Run → Run All Cells** (hoặc nút "Run All" nếu dùng qua VS Code) để chạy tuần tự từ đầu đến cuối — cần thiết khi các ô phía sau phụ thuộc biến đã khai báo ở ô phía trước.

### Lưu ý khi ô có `input()`

- Trong JupyterLab (trình duyệt): ô nhập liệu hiện ngay bên dưới cell đang chạy
- Trong VS Code: ô nhập liệu hiện ở popup nhỏ trên đầu màn hình — vẫn hoạt động đúng, chỉ khác giao diện

### Cài thêm thư viện còn thiếu (nếu báo `ModuleNotFoundError`)

Mở terminal, đảm bảo đã `conda activate ai_prj`, rồi:

```
pip install <ten-thu-vien>
```

Ví dụ: `pip install torch` (cho Lab 02).
