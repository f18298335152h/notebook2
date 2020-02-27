import numpy as np


class Log2_Quant(object):
    def __init__(self):
        pass

    # bits_want_range  : extend or compress origin weight to the range (2**bits - 1) you want to  
    def get_mse(self, bits_want_range=12, feature_range=(0.0, 100.0),
            weight_range=(-1.0, 1.0),
            feature_dim=256,
            output_dim=512):
        np.random.seed(42)

        feature = np.random.uniform(
            low=feature_range[0], high=feature_range[1], size=(feature_dim,))
        weight = np.random.uniform(
            low=weight_range[0], high=weight_range[1],
            size=(feature_dim*output_dim,))
        
        min_wt = weight_range[0]
        max_wt = weight_range[1]

        feature_min = feature[0]
        feature_max = feature[-1]
        
        max_bits = np.log2(max(abs(min_wt),abs(max_wt)))
        # upper int
        max_bits = int(np.ceil(max_bits))
        
        # frac bits
        frac_bits = bits_want_range - max_bits
        quant_wt = np.round(weight*(2**frac_bits))
        quant_wt_rev = quant_wt/(2**frac_bits)



class Linear_Quant(object):
    def __init__(self):
        pass
    
    def Quant(self, Vx, Q, RQM):
        return round(Q * Vx) - RQM

    def QuantRevert(self, VxQuant, Q, RQM):
        return (VxQuant + RQM) / Q

    def ListQuant(self, data_list, quant_bits):
        data_min = min(data_list)
        data_max = max(data_list)

        Q = ((1 << quant_bits) - 1) * 1.0 / (data_max - data_min)
        RQM = (int)(round(Q*data_min))
        
        quant_data_list = []
        for x in data_list:
            quant_data = self.Quant(x, Q, RQM)
            quant_data_list.append(quant_data)
        quant_data_list = np.array(quant_data_list)
        return (Q, RQM, quant_data_list)

    def ListQuantRevert(self, quant_data_list, Q, RQM):
        quant_revert_data_list = []
        for quant_data in quant_data_list:
            revert_quant_data = self.QuantRevert(quant_data, Q, RQM)
            quant_revert_data_list.append(revert_quant_data)
        quant_revert_data_list = np.array(quant_revert_data_list)
        return quant_revert_data_list

    def get_mse(self, quant_bits=12,
            feature_range=(0.0, 100.0),
            weight_range=(-1.0, 1.0),
            feature_dim=256,
            output_dim=512):
        np.random.seed(42)

        feature = np.random.uniform(
            low=feature_range[0], high=feature_range[1], size=(feature_dim,))
        weight = np.random.uniform(
            low=weight_range[0], high=weight_range[1],
            size=(feature_dim*output_dim,))
       
        feature_Q, feature_RQM, feature_quant = self.ListQuant(feature, quant_bits)
        weight_Q, weight_RQM, weight_quant = self.ListQuant(weight, quant_bits)
        
        float_dotprod = np.dot(feature, weight.reshape(feature_dim, output_dim))

        weight_quant = weight_quant + weight_RQM
        quant_dotprod = np.dot(feature_quant+feature_RQM,
            weight_quant.reshape(feature_dim, output_dim))/(feature_Q*weight_Q)
    
        mean_squared_error = ((float_dotprod - quant_dotprod) ** 2).mean()
        print('mean_squared_error = ', mean_squared_error)
        return mean_squared_error

if __name__ == '__main__':
    log2_quant = Log2_Quant()
    log2_quant.get_mse()
