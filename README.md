# Reviews To Stars

My implementation of Kim Yoon’s Convolutional Neural Networks for Sentence Classification.\

<p>The network architecture consists mainly of three blocks:</p>
<ul>
<li>1-First block is for words embedding.</li>
<li>2-Second block is one dimensional CNN with multiple filters to extract several relations that can be found between sentences.</li>
<li>3-Third block is max pooling block and concatenation block to concatenate the features before the softmax.</li> 
</ul>

# Network Architecture
![NW](https://github.com/AhmedFakhry47/Reviews-To-Stars/blob/master/ModelDesign.png)

# Dependencies are: 
<ul>
<li>Pandas</li>
<li>Tensorflow</li>
<li>Numpy</li>
<li>Keras</li>
<li>Sklearn</li>
</ul>

# Model Output
![First](https://github.com/AhmedFakhry47/Reviews-To-Stars/blob/master/Trial1.gif)
![Second](https://github.com/AhmedFakhry47/Reviews-To-Stars/blob/master/Trial2.gif)
