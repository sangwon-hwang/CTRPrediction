testBranch version
Research Collaboration with Nathan Janos(System1) 

We introude time series data augmentation using a few operations including summation, finite difference, and average. In the test, we could find out that a CNN performs better in next min CVR(conversion rate) prediction with 1)our training data set 2)2 dimnesion kernel.

finite difference
<br/>
<img src="/data/cnn01.png" width="450px" height="300px" title="finite difference" alt="RubberDuck">
</img>
<br/>

aggregation
<br/>
<img src="/data/cnn02.png" width="450px" height="300px" title="aggregation" alt="RubberDuck">
</img>
<br/>

final input
<br/>
<img src="/data/cnn07.png" width="450px" height="300px" title="final input" alt="RubberDuck">
</img>
<br/>

As other Neural Processes do, our simple CNN model also shows that lerning rate/scheduler settings are the most important factor as to fitting. We proceeded several test with different lerning schedulers which fit for CVR data.

learning scheduler test
<br/>
<img src="/data/cnn_table.png" width="450px" height="300px" title="learning scheduler test" alt="RubberDuck">
</img>
<br/>

Results
<br/>
<img src="/data/cnn05.png" width="450px" height="300px" title="learning scheduler test" alt="RubberDuck">
</img>
<br/>

<img src="/data/cnn06.png" width="450px" height="300px" title="learning scheduler test" alt="RubberDuck">
</img>
<br/>
