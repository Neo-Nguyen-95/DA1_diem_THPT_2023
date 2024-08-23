# DA1_diem_THPT_2023

## Introduction

Khi kinh tế ngày càng phát triển, khoảng cách giàu - nghèo càng lớn, dẫn đến việc bất bình đẳng trong nhiều lĩnh vực, đặc biệt là giáo dục. Nhưng sự phát triển kinh thế thực chất đang ảnh hưởng đến mảng nào trong giáo dục nhiều nhất? Trong bài phân tích này sẽ chỉ ra những môn học bị ảnh hưởng nhiều nhất bởi sự phân hoá trong phát triển kinh tế.

Báo cáo tập trung phân tích điểm số các môn học và phân bố điểm số của các thí sinh thi tốt nghiệp THPT năm 2023 để chỉ ra sự ảnh hưởng trên.

## Methodology

**Đặc tính của dataset:**

-   Bộ data gồm điểm số của 1022060 thí sinh (TS) tham dự kì thi THPT 2023. Số bao danh được mã hoá với 2 số đầu tiên là mã khu vực, còn 6 số cuối là thứ tự của TS.

-   Ba môn thi bắt buộc là Toán, Ngữ Văn, Ngoại Ngữ.

-   Ba môn thi của khối tự nhiên là Vật Lí, Hoá Học, Sinh Học

-   Ba môn thi của khối xã hội là Lịch Sử, Địa Lí, GDCD.

-   Hai môn thi của khối giáo dục thường xuyên là Lịch Sử, Địa Lí

-   HS sẽ thi ba môn bắt buộc và một tổ hợp môn.

**Các bước làm sạch dữ liệu:**

Chỉ xét điểm của các thí sinh hoàn thành đủ tất cả các bài thi. Với môn Ngoại Ngữ, có 7 mã (từ N1 đến N7) tương ứng với 7 ngôn ngữ khác nhau, nhưng số TS thi Tiếng Anh chiếm 99.44%, nên bài phân tích này chỉ phân tích TS thi Tiếng Anh.

Sau khi làm sạch số lượng TS được thống kê như sau:

-   312364 TS thi khối tự nhiên, chiếm 35.93%

-   555906 TS thi khối xã hội, chiếm 63.95%

-   1014 TS từ khối GDTX, chiếm 0.12%

Vì số lượng TS từ khối GDTX quá ít mà lại có chênh lệch môn học với khối xã hội, lên các TS từ khối này sẽ bị loại bỏ khỏi nghiên cứu.

Như vậy, tổng số TS trong nghiên cứu này là 868270 (bằng 85% tổng số data).
