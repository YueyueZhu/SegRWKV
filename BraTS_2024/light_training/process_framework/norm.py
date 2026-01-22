


from light_training.preprocessing.normalization.default_normalization_schemes import CTNormalization, CTNormStandard


                                                            
def norm_func(data, seg=None, **kwargs):
    normalizer = CTNormStandard(a_min=-175, 
                                    a_max=250, 
                                    b_min=0.0,
                                    b_max=1.0, clip=True)

    data = normalizer(data)

    return data 
