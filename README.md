TL,DR : Bài viết sau đây là hướng dẫn cơ bản về deep learning trên Torch để huấn luyện với dữ liệu MNIST.  
Để tải dữ liệu (file dạng .t7), chạy dòng 
```Shell
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar -xzf mnist.t7.tgz
rm mnist.t7.tgz
mv mnist.t7 data
```

Để chạy chương trình, 
```Shell
th train.lua
```
Sau đây, chúng ta sẽ đi vào chi tiết việc huấn luyện MNIST.  

1. Tải dữ liệu  
  - Có nhiều cách để tải dữ liệu. Mình chọn nguồn dữ liệu đã được tải sẵn vào file t7. File dạng t7 là một dạng file tương tự như 
  json để giúp việc đọc / ghi dữ liệu trên torch.  
  
2. Preparation (including setting up the model and other things)
3. Training
4. Evaluation

