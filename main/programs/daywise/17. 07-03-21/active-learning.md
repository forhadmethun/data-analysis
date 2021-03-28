- From dataset, we collect `data sample labels` and train a `init model` 
- Unlabelled data are sent to the model and based on `confidence` either it gives label(for high confidence) or ask oracle for decision
    - oracle is someone who is an expert with database
    - ? How to decide confidence
        - some assumptions: 
            - minifold *
            - need to study *
    
### Paper: A survey on instance selection for active learning
#### Two category
 - uncertainty of independent and identical distributed instances
 - active learning by further taking into account instances

=> classification & clustering

=> For small labelled data we can have the following approach
 - semi-supervised : utilize unlabelled data & by taking geometry of data distribution such as cluster & manifolds 
 - active learning : `selectively labels instances` by `interactively selecting` most informative instances based on certain `instance-selection criteria`

Paper goal, survey on active learning from an instance-selection perspective, where achieving high performance by few labels instances as possible

There are many ways for assessing uncertainty criteria, from instance-selection perspective two category
 - Utility metrics based on uncertainty of IID instances
    - treat samples as independent and identically distributed instances, might select similar instances result in redundancy
 - Utility metrics further taking into account instance `correlation`
    - utilizes some similarity measures to discriminate differences between instances
