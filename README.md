# Name-Generator
The project is an LSTM model trained on a small dataset of 3000 names. Model generates names from model by selecting one out of top 3 letters suggested by model at a time until an EOS (End Of Sentence) character is not encountered. <br>
<b>Input for the model is any letter than you want the suggested names to start from.<b><br>
Output are the 20 names generated by model.<br>

<h3>Epoch vs Loss Graph<h3>
  
![Capture1](https://user-images.githubusercontent.com/22273562/146696231-b67cb264-7dfc-48a8-a0cd-3b8a52ab8319.PNG)

<h3> 20 names generated by model for input of letter 'a' and 'x'<h3>
  
![Capture](https://user-images.githubusercontent.com/22273562/146696301-5a1a6dc3-3c36-4f7d-8549-70edd5ca0268.PNG)
