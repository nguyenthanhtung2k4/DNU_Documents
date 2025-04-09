import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Định nghĩa đường dẫn file gốc và file đã dịch
input_file = "SuperMarket Analysis.csv"  # Đổi thành đường dẫn file của bạn
output_file = "SuperMarket_Analysis_Vietnamese.csv"

# Đọc file CSV
df = pd.read_csv(input_file)

# Định nghĩa mapping đổi tên cột
column_mapping = {
    "Invoice ID": "Mã hóa đơn",
    "Branch": "Chi nhánh",
    "City": "Thành phố",
    "Customer type": "Loại khách hàng",
    "Gender": "Giới tính",
    "Product line": "Ngành hàng",
    "Unit price": "Giá đơn vị",
    "Quantity": "Số lượng",
    "Tax 5%": "Thuế 5%",
    "Sales": "Doanh thu",
    "Date": "Ngày",
    "Time": "Giờ",
    "Payment": "Phương thức thanh toán",
    "cogs": "Giá vốn hàng bán",
    "gross margin percentage": "Tỷ suất lợi nhuận gộp",
    "gross income": "Lợi nhuận gộp",
    "Rating": "Đánh giá"
}

# Áp dụng thay đổi tên cột
df_vietnamese = df.rename(columns=column_mapping)

# Chuyển cột ngày sang định dạng datetime
df_vietnamese["Ngày"] = pd.to_datetime(df_vietnamese["Ngày"], format="%m/%d/%Y")

# Lưu file CSV mới
df_vietnamese.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"File đã được lưu: {output_file}")

# 1. Tổng doanh thu và lợi nhuận
total_sales = df_vietnamese["Doanh thu"].sum()
total_profit = df_vietnamese["Lợi nhuận gộp"].sum()
print(f"Tổng doanh thu: {total_sales}")
print(f"Tổng lợi nhuận: {total_profit}")

# 2. Phân tích theo ngành hàng
plt.figure(figsize=(10,5))
sns.barplot(x=df_vietnamese["Ngành hàng"], y=df_vietnamese["Doanh thu"], estimator=sum)
plt.xticks(rotation=45)
plt.title("Doanh thu theo ngành hàng")
plt.show()

# 3. Phân tích số lượng khách hàng theo loại khách hàng
plt.figure(figsize=(6,4))
sns.countplot(x=df_vietnamese["Loại khách hàng"])
plt.title("Phân bố loại khách hàng")
plt.show()

# 4. Phân tích thời gian cao điểm
plt.figure(figsize=(10,5))
df_vietnamese["Giờ giao dịch"] = pd.to_datetime(df_vietnamese["Giờ"], format="%I:%M:%S %p").dt.hour
sns.countplot(x=df_vietnamese["Giờ giao dịch"], palette="viridis")
plt.title("Số lượng giao dịch theo giờ trong ngày")
plt.show()

# 5. Phân tích phương thức thanh toán
plt.figure(figsize=(6,4))
sns.countplot(x=df_vietnamese["Phương thức thanh toán"])
plt.title("Phân bố phương thức thanh toán")
plt.show()

# 6. Hồi quy tuyến tính để phân tích tác động của giá và số lượng lên doanh thu
X = df_vietnamese[["Giá đơn vị", "Số lượng"]]  # Biến độc lập
X = sm.add_constant(X)  # Thêm hệ số chặn
Y = df_vietnamese["Doanh thu"]  # Biến phụ thuộc

# Xây dựng mô hình hồi quy
model = sm.OLS(Y, X).fit()
print(model.summary())

# 7. Tính toán hiệu quả khuyến mãi
# Giả định: Khuyến mãi là mức giảm giá trên đơn vị sản phẩm
# Tạo một cột mới giả lập giá giảm 10%
df_vietnamese["Giá sau khuyến mãi"] = df_vietnamese["Giá đơn vị"] * 0.9

# Tính doanh thu dự kiến sau khuyến mãi
df_vietnamese["Doanh thu sau khuyến mãi"] = df_vietnamese["Giá sau khuyến mãi"] * df_vietnamese["Số lượng"]

# So sánh doanh thu trước và sau khuyến mãi
discount_effect = df_vietnamese[["Ngành hàng", "Doanh thu", "Doanh thu sau khuyến mãi"]].groupby("Ngành hàng").sum()
print(discount_effect)

# 8. Dự đoán doanh thu trong tương lai
# Chọn các biến đầu vào
X_future = df_vietnamese[["Giá đơn vị", "Số lượng"]]
Y_future = df_vietnamese["Doanh thu"]

# Tách tập dữ liệu thành train và test
X_train, X_test, Y_train, Y_test = train_test_split(X_future, Y_future, test_size=0.2, random_state=42)

# Huấn luyện mô hình hồi quy tuyến tính
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Dự đoán doanh thu
Y_pred = regressor.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Dự đoán doanh thu tương lai với giá và số lượng trung bình
future_data = np.array([[df_vietnamese["Giá đơn vị"].mean(), df_vietnamese["Số lượng"].mean()]])
predicted_sales = regressor.predict(future_data)
print(f"Dự đoán doanh thu tương lai: {predicted_sales[0]}")
