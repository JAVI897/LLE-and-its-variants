# LLE (Locally linear embedding) and its variants

Locally Linear Embedding is another method for non-linear dimensionality reduction. It was proposed by Roweis and Saul in 2000. At that time LLE was a novel techinque. Earlier techniques for non linear dimensionality reduction were based on MDS to preserve euclidean distances or more sophisticated distances such as geodesic distances (ISOMAP). LLE takes a different approach, LLE won't preserve distances, but local geometry of the data.

In this project I'll be publisihng some implementations for new variants of LLE.  In addition, I'll reproduce results of the paper on which the algorithm is based. However,  implementations will not be computationally optimal, this repository is primarily for research purposes.

### Implementations

* __LLE__: 

  - Implemented following [1] For a more detailed description on how the algorithm works see; [Think Globally, Fit Locally; LLE](https://javi897.github.io/LLE/)

    Example; **LLE - Swiss Roll**

<img src="https://github.com/JAVI897/LLE-and-its-variants/blob/master/images/LLE-Swiss-roll.png" style="zoom: 70%;" />

- __ISOLLE__: 

  - In LLE, each data point is reconstructed from a linear combination of its n nearest neighbors, which are typically found using the Euclidean distance. In ISOLLE the search for the neighbors is performed with respect to the geodesic distance. This leads to a more accurate preservation of the data structure [2]

  Example; **ISOLLE - Swiss Roll**
  
  <img src="https://github.com/JAVI897/LLE-and-its-variants/blob/master/images/ISOLLE-Swiss-roll.png" style="zoom:70%;" />

### References

[1]  S. T. Roweis and L. K. Saul. Think Globally, Fit Locally: Unsupervised Learning of Low Dimensional Manifolds. Journal of Machine Learning Research 4 (2003) 119-155

[2]  Varini, Claudio & Degenhard, Andreas & Nattkemper, Tim. (2005). ISOLLE: Locally linear embedding with geodesic distance. 331-342. 10.1007/11564126_34. 

