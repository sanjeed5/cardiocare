# CardioCare

Developed as part of Intel Python HACKFURY2

## Overview of project

Cardiovascular diseases, are among the most widespread and costly health problems faced by the nation today. It is critical to address the risk factors as early as possible to prevent potential complications of chronic cardiovascular diseases. 
Being undergraduates who would be specializing in Biomedical Design in our dual degree programme, we chose this project to provide a solution that would :
-be an easy-to-use application 
-which can be used by general physicians to receive ML-based diagnosis with high accuracy
-positively help to warn patients of any impending heart disease complications, well in advance. 
-help control the risk factors for possible cardiovascular diseases.

The deployed app prototype uses Intel® Distribution for Python, Intel numpy and Intel scikit-learn packages. From our research, we understood that these are much faster than the packages we previously used. The speed boost was achieved using Intel® Data Analytics Acceleration Library through daal4py package. We also plan to use Intel® Optimization for TensorFlow in the final product.

We aspire to collaborate with a cardiology-based hospital or medical institute, use their larger database and datasets and utilize Intel® Optimization for Tensorflow to create a much more accurate model using deep neural networks. This would help us in widening the capability of our model to predict disease and disease risk of the patient along with the expertise of the cardiologist.

Our final app would be an easy-to-use web app which a general physician can use to understand cardiovascular risks. We will be working with cardiologists to better understand the parameters and provide more accurate results.

Our vision:

A combination of poverty, ignorance, lack of access to quality care and smoking may be driving heart-disease related deaths in India, a study shows[1](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(18)30242-0/fulltext). And the alarming fact is that the study shows that deaths due to coronary heart diseases and strokes were more common among the urban population at the turn of the century, but the trend has reversed since then. Rate of mortality due to coronary heart diseases increased among rural men by over 40% even as it declined among urban men. For females, the increase was over 56% in rural India[2](https://www.livemint.com/Politics/fKmvnJ320JOkR7hX0lbdKN/Rural-India-surpasses-urban-in-heart-diseaserelated-deaths.html).

Our vision is to eliminate this disparity and provide access to quality care to everybody. The preliminary diagnosis with our app, will inform them of the gravity of the impending cardiovascular risks and they can eliminate the risks at an early stage.

After we deploy CardioCare succesfully, we aspire to expand the prediction to diseases concerning other vital organs of the body using Medical Image Analysis and Deep Learning and make this a full-fledged health risks prediction app.


References:
1. https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(18)30242-0/fulltext
2. https://www.livemint.com/Politics/fKmvnJ320JOkR7hX0lbdKN/Rural-India-surpasses-urban-in-heart-diseaserelated-deaths.html



### Access the website [cardiocare.herokuapp.com](http://cardiocare.herokuapp.com)

### Or,

- git clone https://github.com/sanjeed5/cardiocare
- cd cardiocare
- flask run
