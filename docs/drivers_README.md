## Supported Autonomous Driving Agents
1. Transfuser:

    For older GPUs, cuda 10.2 is available as precompiled binary. Transfuser 
dependencies can be directly installed. For newer GPUs, check setup notes.

    Transfuser also uses pretrained image models from ```timm```. They are
automatically downloaded on their first use. 
Image classification model used in the publicly available pretrained transfuser 
model: regnety_032_ra

2. Interfuser

    timm version installed for Transfuser may not work for Interfuser. In that
case copy the timm folder from Interfuser github repository and change the 
import statements in the dependent files accordingly.

