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
  json để giúp việc đọc / ghi dữ liệu trên torch và có hỗ trợ việc đọc / ghi với Lua khá nhanh.   
  
2. Khâu chuẩn bị (bao gồm khởi tạo model, hàm mất mát, và ma trận lỗi)   
  - Khá giống với PyTorch, Torch đã có hỗ trợ khá nhiều layers trong deep learning. Ở đây mình chỉ dùng hai convolutional layers (đi kèm với activation function và max pooling layers). Sau đó mình sử dụng tiếp hai hàm fully connected layers (cũng bao gồm làm activation function). Để ý là ở layer cuối cùng, mình cho output là số lượng các classes và dùng hàm LogSoftMax. Các bạn có thể tìm phần setup ở hàm setup_model() trong code.   
  - Hàm mất mát mình chọn khá thông dụng là hàm Negative Log-Likelihood. Hàm này thường được sử dụng khi layer cuối cùng của model là LogSoftmax.  
  - Phần cuối là thêm vào ma trận lỗi (confusion matrix). Phần này bạn có thể biết được model của bạn đang có hiệu quả như thế nào sau mỗi lần chạy evaluation.   
  - Mình có thêm vào phần sử dùng cuda / cudnn hay không. Ở Torch thì phần chuyển từ model sử dụng khi train trên CPU và GPU được chuyển đổi rất đơn giản.   
3. Training
  - Để train theo batch, bạn chỉ cần để input có định dạng là batch_size * data, và output là batch_size * target.   
  - Cách thức train trên torch khá đơn giản:
  ```Lua
      local output = net:forward(input)
      local f = criterion:forward(output, target)
      loss = loss + (f / batch_size)

      net:zeroGradParameters()

      local gradients = criterion:backward(output, target)
      net:backward(input, gradients)
      net:updateParameters(learning_rate)
   ```   
   - Các bạn để ý là hàm loss mình tính vào thật ra không cần thiết, nhưng nhiều khi các bạn sẽ cần sử dụng nó để kiểm tra xem model có đang thật sự thực hiện ý bạn muốn hay không. Một cách kiểm tra mình hay dùng là in ra hàm loss sau một thời gian training ngắn để xem hàm loss có giảm đi hay không.   
4. Evaluation
   - Để chạy kiểm tra test set thì cũng giống như lúc chạy  training, nhưng bạn chỉ cần giữ lại phần output và đưa vào confusion matrix. Khi bạn in ra confusion matrix, độ chính xác sẽ được hiển thị ra tự động.   

