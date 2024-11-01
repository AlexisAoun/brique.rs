# Structure of the CSV file - expected_params : 

5 Epochs, 1 Epoch => 21 matrix :

Matrix 1 to 6 = weights and biases, alternated. W,b,W,b

Matrix 7 to 9 = intermadiate scores of the evaluation y1 y2 y3

Matrix 10 scores after exp()

Matrix 11 final softmax result 

Matrix 12 gradient of the loss, d_score

Matrix 13 to 20 alternates between various gradients of the backprog => dW3, dB3, dZ2, DW2, DB2, dZ1, dW1, dB1 

Matrix 21 contains the loss of the interation => [data_loss, reg_loss, total_loss]

Same pattern repeats 5 times for each epoch 

