import numpy as np
from scipy.optimize import minimize, root_scalar, Bounds
from numpy.polynomial.hermite import hermfit, hermval, hermder
import copy

# Conversions between ZCB prices, spot rates forward rates and libor rates
def zcb_prices_from_spot_rates(T,R,method = "continuous"):
    M = len(T)
    p = np.zeros([M])
    if method == "continuous":
        for i in range(0,M):
            if T[i] < 1e-10:
                p[i] = 1
            else:
                p[i] = np.exp(-R[i]*T[i])
    elif method == "simple":
        for i in range(0,M):
            if T[i] < 1e-10:
                p[i] = 1
            else:
                p[i] = 1/(1+R[i]*T[i])
    elif method == "discrete":
        for i in range(0,M):
            if T[i] < 1e-10:
                p[i] = 1
            else:
                p[i] = 1/(1+R[i])**T[i]
    return p

def spot_rates_from_zcb_prices(T,p,method = "continuous"):
    M = len(T)
    R = np.zeros([M])
    if method == "continuous":
        for i in range(0,M):
            if T[i] < 1e-10:
                R[i] = np.nan
            else:
                R[i] = -np.log(p[i])/T[i]
    elif method == "simple":
        for i in range(0,M):
            if T[i] < 1e-10:
                R[i] = np.nan
            else:
                R[i] = (1-p[i])/(T[i]*p[i])
    elif method == "discrete":
        for i in range(0,M):
            if T[i] < 1e-10:
                R[i] = np.nan
            else:
                R[i] = p[i]**(-1/T[i]) - 1
    return R

def forward_rates_from_zcb_prices(T,p,horizon = 1,method = "continuous"):
    # horizon = 0 corresponds to approximated instantaneous forward rates.
    M = len(T)
    f = np.nan*np.ones([M])
    if method == "continuous":
        if horizon == 0:
            f[0] = (np.log(p[0])-np.log(p[1]))/(T[1]-T[0])
            f[-1] = (np.log(p[-2])-np.log(p[-1]))/(T[-1]-T[-2])
            i = 1
            while i < M - 1.5:
                f[i] = (np.log(p[i-1])-np.log(p[i+1]))/(T[i+1]-T[i-1])
                i += 1
        elif 0 < horizon:
            i = horizon
            while i < M - 0.5:
                f[i] = (np.log(p[i-horizon])-np.log(p[i]))/(T[i]-T[i-horizon])
                i += 1
    elif method == "simple":
        if horizon == 0:
            f[0] = (p[0] - p[1])/(p[1]*(T[1]-T[0]))
            f[-1] = (p[-2] - p[-1])/(p[-1]*(T[-1]-T[-2]))
            i = 1
            while i < M - 1.5:
                f[i] = (p[i-1] - p[i+1])/(p[i+1]*(T[i+1]-T[i-1]))
                i += 1
        elif 0 < horizon:
            i = horizon
            while i < M - 0.5:
                f[i] = (p[i-horizon] - p[i])/(p[i]*(T[i]-T[i-horizon]))
                i += 1
    elif method == "discrete":
        if horizon == 0:
            f[0] = (p[0]/p[1])**(1/(T[1]-T[0])) - 1
            f[-1] = (p[-2]/p[-1])**(1/(T[-1]-T[-2])) - 1
            i = 1
            while i < M - 1.5:
                f[i] = (p[i-1]/p[i+1])**(1/(T[i+1]-T[i-1])) - 1
                i += 1
        elif 0 < horizon:
            i = horizon
            while i < M - 0.5:
                f[i] = (p[i-horizon]/p[i])**(1/(T[i]-T[i-horizon])) - 1
                i += 1
    return f

# Annuity Bond
def alpha_hage(T,R,K,coupon_freq):
    # T: Maturity
    # R: Coupon Rate
    # K: Principal
    # coupon_freq: Frequency of Coupons ("quarterly", "semiannual" or "annual")
    if coupon_freq == "quarterly":
        alpha = 0.25
    elif coupon_freq == "semiannual":
        alpha = 0.5
    elif coupon_freq == "annual":
        alpha = 1
    N = T/alpha
    alpha_hage = ((1+alpha*R)**N-1)/(alpha*R*(1+alpha*R)**N)
    return alpha_hage

# Serial Bond
def serial_bond_cashflows(T,R,K,coupon_freq):
    # T: Maturity
    # R: Coupon Rate
    # K: Principal
    # coupon_freq: Frequency of Coupons ("quarterly", "semiannual" or "annual")
    if coupon_freq == "quarterly":
        alpha = 0.25
    elif coupon_freq == "semiannual":
        alpha = 0.5
    elif coupon_freq == "annual":
        alpha = 1
    N = int(round(T/alpha)) - 1
    cf, interest, principal = np.zeros([N+1]), np.zeros([N+1]), np.zeros([N+1])
    principal[0] = K
    for n in range(1,N+1):
        interest[n] = principal[n-1]*alpha*R
        principal[n] = principal[n-1] - K/N
        cf[n] = interest[n] + K/N
    return cf, interest, principal

# Fixed rate bond
def price_fixed_rate_bond(t,R_coupon,maturity,principal,coupon_freq,T,p):
    # t:=           Present time (usually set to t=0) in years
    # R_coupon:=           Annualixzed simple coupon rate
    # maturity:=           Maturity of the bond in years
    # prin :=          Principal
    # coupon_freq := Frequency of coupon paymenys ("annual", "semiannual", "quarterly")
    # T :=          Maturities of the zero coupon bond prices used when pricing the fixed rate coupon bond
    # p :=          Zero coupon bond prices corresponding to T
    # NOTE! If t is a coupon date, it is assumed that the coupon at t should NOT be included in the price of the fixed rate coupon bond.
    T_coupon = []
    if coupon_freq == "quarterly":
        alpha = 0.25
        for n in range(1,int(4*maturity)+1):
            if t < n*alpha <= maturity:
                T_coupon.append(n*alpha)
    elif coupon_freq == "semiannual":
        alpha = 0.5
        for n in range(1,int(2*maturity)+1):
            if t < n*alpha <= maturity:
                T_coupon.append(n*alpha)
    elif coupon_freq == "annual":
        alpha = 1
        for n in range(1,int(maturity)+1):
            if t < n*alpha <= maturity:
                T_coupon.append(n*alpha)
    p_coupon = for_values_in_list_find_value_return_value(T_coupon,T,p)
    price = 0
    for i in range(0,len(p_coupon)):
        price += p_coupon[i]*alpha*R_coupon*principal
    price += p_coupon[-1]*principal
    return price

# List operations
def find_value_return_value(val,L1,L2,precision = 10e-8):
    # This function searches for 'val' in 'L1' and returns index 'idx' of 'val' in 'L1' and 'L2[idx]'.
    Ind, output = False, []
    for idx, item in enumerate(L1):
        if abs(val-item) < precision:
            Ind = True
            output.append((idx,L2[idx]))
    return Ind, output

def for_values_in_list_find_value_return_value(L1,L2,L3,precision = 10e-8):
    # For all 'item' in L1, this function searches for 'item' in L2 and returns the value corresponding to same index from 'L3'.
    if type(L1) == int or type(L1) == float or type(L1) == np.float64 or type(L1) == np.int32 or type(L1) == np.int64:
        output = None
        Ind, output_temp = find_value_return_value(L1,L2,L3,precision)
        if Ind == True:
            output = output_temp[0][1]
    elif type(L1) == tuple or type(L1) == list or type(L1) == np.ndarray:
        output = len(L1)*[None]
        for i, item in enumerate(L1):
            Ind, output_temp = find_value_return_value(item,L2,L3,precision)
            if Ind == True:
                output[i] = output_temp[0][1]
    return output

# ZCB curvefitting
def zcb_curve_fit(data_input,interpolation_options = {"method": "linear"},scaling = 1):
    data = copy.deepcopy(data_input)
    data_known = []
    libor_data, fra_data, swap_data = [], [], []
    # Separateing the data and constructing data_known from fixings
    for item in data:
        if item["instrument"] == "libor":
            libor_data.append(item)
            data_known.append({"maturity":item["maturity"],"rate":np.log(1+item["rate"]*item["maturity"])/item["maturity"]})
        elif item["instrument"] == "fra":
            fra_data.append(item)
        elif item["instrument"] == "swap":
            swap_data.append(item)
    # Adding elements to data_knwon based on FRAs
    I_done = False
    while I_done == False:
        for fra in fra_data:
            I_exer, known_exer = value_in_list_of_dict_returns_I_idx(fra["exercise"],data_known,"maturity")
            I_mat, known_mat = value_in_list_of_dict_returns_I_idx(fra["maturity"],data_known,"maturity")
            if I_exer == True and I_mat == False:
                data_known.append({"maturity":fra["maturity"],"rate":(known_exer["rate"]*known_exer["maturity"]+np.log(1+(fra["maturity"]-fra["exercise"])*fra["rate"]))/fra["maturity"]})
                I_done = False
                break
            if I_exer == False and I_mat == True:
                pass
            if I_exer == True and I_mat == True:
                pass
            else:
                I_done = True
    T_known, T_swap, T_knot = [], [], []
    R_known = []
    # Finding T's and corresponding R's where there is some known data
    for known in data_known:
        T_known.append(known["maturity"])
        R_known.append(known["rate"])
    # Finding T_swap - The times where there is a cashflow to at least one of the swaps.
    for swap in swap_data:
        T_knot.append(swap["maturity"])
        if swap["float_freq"] == "quarterly":
            if value_in_list_returns_I_idx(0.25,T_known)[0] == False and value_in_list_returns_I_idx(0.25,T_swap)[0] == False:
                T_swap.append(0.25)
        elif swap["float_freq"] == "semiannual":
            if value_in_list_returns_I_idx(0.5,T_known)[0] == False and value_in_list_returns_I_idx(0.5,T_swap)[0] == False:
                T_swap.append(0.5)
        elif swap["float_freq"] == "annual":
            if value_in_list_returns_I_idx(1,T_known)[0] == False and value_in_list_returns_I_idx(1,T_swap)[0] == False:
                T_swap.append(1)
        if swap["fixed_freq"] == "quarterly":
            for i in range(1,4*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.25,T_known)[0] == False and value_in_list_returns_I_idx(i*0.25,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.25,T_swap)[0] == False:
                    T_swap.append(i*0.25)
        elif swap["fixed_freq"] == "semiannual":
            for i in range(1,2*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.5,T_known)[0] == False and value_in_list_returns_I_idx(i*0.5,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.5,T_swap)[0] == False:
                    T_swap.append(i*0.5)
        elif swap["fixed_freq"] == "annual":
            for i in range(1,swap["maturity"]):
                if value_in_list_returns_I_idx(i,T_known)[0] == False and value_in_list_returns_I_idx(i*1,T_knot)[0] == False and value_in_list_returns_I_idx(i,T_swap)[0] == False:
                    T_swap.append(i)
    # Finding T_fra and T_endo
    T_endo, T_fra = [], []
    fra_data.reverse()
    for fra in fra_data:
        if value_in_list_returns_I_idx(fra["maturity"],T_known)[0] == False:
            I_fra_mat, idx_fra_mat = value_in_list_returns_I_idx(fra["maturity"],T_fra)
            I_endo_mat, idx_endo_mat = value_in_list_returns_I_idx(fra["maturity"],T_endo)
            if I_fra_mat is False and I_endo_mat is False:
                T_fra.append(fra["maturity"])
            elif I_fra_mat is True and I_endo_mat is False:
                pass
            elif I_fra_mat is False and I_endo_mat is True:
                pass
            elif I_fra_mat is True and I_endo_mat is True:
                T_fra.pop(idx_fra_mat)
        if value_in_list_returns_I_idx(fra["exercise"],T_known)[0] == False:
            I_fra_exer, idx_fra_exer = value_in_list_returns_I_idx(fra["exercise"],T_fra)
            I_endo_exer, idx_endo_exer = value_in_list_returns_I_idx(fra["exercise"],T_endo)
            if I_fra_exer is False and I_endo_exer is False:
                T_endo.append(fra["exercise"])
            elif I_fra_exer is True and I_endo_exer is False:
                T_fra.pop(idx_fra_exer)
                T_endo.append(fra["exercise"])
            elif I_fra_exer is False and I_endo_exer is True:
                pass
            elif I_fra_exer is True and I_endo_exer is True:
                T_fra.pop(idx_fra_exer)
    fra_data.reverse()
    # Fitting the swap portion of the curve
    T_swap_fit = T_known + T_swap + T_knot
    T_swap_fit.sort(), T_fra.sort(), T_endo.sort()
    R_knot_init = [None]*len(swap_data)
    for i, swap in enumerate(swap_data):
        indices = []
        if swap["fixed_freq"] == "quarterly":
            for j in range(1,4*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.25,T_swap_fit)[1])
        elif swap["fixed_freq"] == "semiannual":
            for j in range(1,2*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.5,T_swap_fit)[1])
        elif swap["fixed_freq"] == "annual":
            for j in range(1,swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j,T_swap_fit)[1])
        swap["indices"] = indices
        R_knot_init[i] = swap["rate"]
        i += 1
    args = (T_known,T_knot,T_swap_fit,R_known,swap_data,interpolation_options,1)
    result = minimize(zcb_curve_swap_fit_obj,R_knot_init,method = 'nelder-mead',args = args,options={'xatol': 1e-12,'disp': False})
    # print(f"Output from the swap portion fit")
    # print(result)
    T_swap_curve, R_swap_curve = T_known + T_knot, R_known + list(result.x)
    T_fra_fit = T_swap_curve + T_fra + T_endo
    T_fra_fit.sort()
    R_fra_fit, R_fra_fit_deriv = interpolate(T_fra_fit,T_swap_curve,R_swap_curve,interpolation_options)
    R_fra_init = [None]*len(T_fra)
    for i in range(0,len(T_fra)):
        R_fra_init[i] = R_fra_fit[value_in_list_returns_I_idx(T_fra[i],T_fra_fit)[1]]
    args = (T_fra,T_known,T_endo,T_fra_fit,R_fra_fit,fra_data,interpolation_options,scaling)
    result = minimize(zcb_curve_fra_fit_obj,R_fra_init,method = 'nelder-mead',args = args,options={'xatol': 1e-12,'disp': False})
    # print(f"Output from the FRA portion fit")
    # print(result)
    R_fra = list(result.x)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra_fit)):
        I_fra, idx_fra = value_in_list_returns_I_idx(T_fra_fit[i],T_fra)
        if I_fra is True:
            R_fra_fit[i] = R_fra[idx_fra]
        elif I_fra is False:
            I_endo, idx_endo = value_in_list_returns_I_idx(T_fra_fit[i],T_endo)
            if I_endo is True:
                R_fra_fit[i] = R_endo[idx_endo]
    return np.array(T_fra_fit), np.array(R_fra_fit)

def zcb_curve_interpolate(T_inter,T,R,interpolation_options = {"method":"linear"}):
    N = len(T_inter)
    p_inter = np.ones([N])
    R_inter = np.zeros([N])
    f_inter = np.zeros([N])
    R_inter, R_inter_deriv = interpolate(T_inter,T,R,interpolation_options = interpolation_options)
    for i in range(0,N):
        f_inter[i] = R_inter[i] + R_inter_deriv[i]*T_inter[i]
        p_inter[i] = np.exp(-R_inter[i]*T_inter[i])
    return p_inter, R_inter, f_inter, T_inter

def extrapolate(x_extra,x,y,extrapolation_options = {"method":"linear"}):
    # Extrapoltion of value corresponding to a choice of x_extra
    if extrapolation_options["method"] == "linear":
        if x_extra < x[0]:
            a = (y[1]-y[0])/(x[1]-x[0])
            b = y[0]-a*x[0]
            y_extra = a*x_extra + b
            y_extra_deriv = a
        elif x[-1] < x_extra:
            a = (y[-1]-y[-2])/(x[-1]-x[-2])
            b = y[-1]-a*x[-1]
            y_extra = a*x_extra + b
            y_extra_deriv = a
        else:
            print(f"WARNING! x_extra is inside the dataset")
    elif extrapolation_options["method"] == "hermite":
        if x_extra < x[0]:
            coefs = hermfit(x[0:extrapolation_options["degree"]+1],y[0:extrapolation_options["degree"]+1],extrapolation_options["degree"])
            y_extra, y_extra_deriv = hermval(x_extra,coefs), hermval(x_extra,hermder(coefs))
        elif x[-1] < x_extra:
            coefs = hermfit(x[-extrapolation_options["degree"]-1:],y[-extrapolation_options["degree"]-1:],extrapolation_options["degree"])
            y_extra, y_extra_deriv = hermval(x_extra,coefs), hermval(x_extra,hermder(coefs))
        else:
            print(f"WARNING! x_extra is inside the dataset")
    elif extrapolation_options["method"] == "nelson_siegel":
        if x_extra < x[0]:
            x1, x2 = x[1]-x[0], x[2]-x[0]
            coefs = nelson_siegel_coef(x1,x2,y[0],y[1],y[2])
            y_extra, y_extra_deriv = coefs[0]+coefs[1]*np.exp(-coefs[2]*(x_extra-x[0])), -coefs[1]*coefs[2]*np.exp(-coefs[2]*(x_extra-x[0]))
        elif x[-1] < x_extra:
            x1, x2 = x[-2]-x[-3], x[-1]-x[-3]
            coefs = nelson_siegel_coef(x1,x2,y[-3],y[-2],y[-1])
            y_extra, y_extra_deriv = coefs[0]+coefs[1]*np.exp(-coefs[2]*(x_extra-x[-3])), -coefs[1]*coefs[2]*np.exp(-coefs[2]*(x_extra-x[-3]))
        else:
            print(f"WARNING! x_extra is inside the dataset")
    return y_extra, y_extra_deriv

def interpolate(x_inter,x,y,interpolation_options = {"method":"linear", "transition": None}):
    N, M = len(x_inter), len(x)
    y_inter, y_inter_deriv = np.nan*np.ones([N]), np.nan*np.ones([N])
    if interpolation_options["method"] == "linear":
        coefs = np.nan*np.ones([M,2])
        for m in range(0,M-1):
            coefs[m,1] = (y[m+1]-y[m])/(x[m+1]-x[m])
            coefs[m,0] = y[m]-coefs[m,1]*x[m]
        coefs[M-1,1] = (y[M-1] - y[M-2])/(x[M-1]-x[M-2])
        coefs[M-1,0] = y[M-1] - coefs[M-1,1]*x[M-1]
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    if idx == 0:
                        y_inter_deriv[n] = coefs[0,1]
                    elif idx == M-1:
                        y_inter_deriv[n] = coefs[M-1,1]
                    else:
                        y_inter_deriv[n] = 0.5*coefs[idx-1,1] + 0.5*coefs[idx,1]
                        y_inter_deriv[n] = coefs[idx,1]
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if interpolation_options["transition"] == "smooth":
                        w_before = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                        y_before, y_after = coefs[idx_before,0] + coefs[idx_before,1]*x_inter[n], coefs[idx_after,0] + coefs[idx_after,1]*x_inter[n]

                        y_inter[n] = w_before*y_before + (1-w_before)*y_after
                        y_inter_deriv[n] = w_before*coefs[idx_before,1] + (1-w_before)*coefs[idx_after,1]
                    else:
                        y_inter[n] = coefs[idx_before,0] + coefs[idx_before,1]*x_inter[n]
                        y_inter_deriv[n] = coefs[idx_before,1]
    elif interpolation_options["method"] == "hermite":
        coefs = np.nan*np.ones([M,interpolation_options["degree"]+1])
        degrees = np.ones(M, dtype = "int")
        for m in range(0, M-1):
            left = min(int(interpolation_options["degree"]/2),m)
            right = min(M-1-m,int((interpolation_options["degree"]+1)/2))
            degrees[m] = left + right
            coefs[m,0:left+right+1] = hermfit(x[m-left:m+right+1],y[m-left:m+right+1],degrees[m])
        coefs[M-1], degrees[M-1] = coefs[M-2], degrees[M-2]
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                y_inter[n], y_inter_deriv[n] = extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx,0:degrees[idx]+1]))
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if interpolation_options["transition"] == "smooth":
                        w_before = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                        y_before = hermval(x_inter[n],coefs[idx_before,0:degrees[idx_before]+1])
                        y_after = hermval(x_inter[n],coefs[idx_after,0:degrees[idx_after]+1])
                        y_inter[n] = w_before*y_before + (1-w_before)*y_after
                        y_inter_deriv[n] = w_before*hermval(x_inter[n],hermder(coefs[idx_before,0:degrees[idx_before]+1])) + (1-w_before)*hermval(x_inter[n],hermder(coefs[idx_after,0:degrees[idx_after]+1]))
                    else:
                        y_inter[n] = hermval(x_inter[n],coefs[idx_before,0:degrees[idx_before]+1])
                        y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx_before,0:degrees[idx_before]+1]))
    elif interpolation_options["method"] == "nelson_siegel":
        coefs, type_fit = np.nan*np.ones([M,3]), M*[None]
        for m in range(1,M-1):
            if (y[m] > y[m-1] and y[m+1] > y[m]) or (y[m] < y[m-1] and y[m+1] < y[m]):
                coefs[m,0:3] = nelson_siegel_coef(x[m]-x[m-1],x[m+1]-x[m-1],y[m-1],y[m],y[m+1])
                type_fit[m] = "nelson_siegel"
            else:
                coefs[m,0:3] = hermfit(x[m-1:m+2],y[m-1:m+2],2)
                type_fit[m] = "hermite"
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                y_inter[n], y_inter_deriv[n] = extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    if idx == 0:
                        if type_fit[idx+1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx]
                            y_inter_deriv[n] = -coefs[idx+1,1]*coefs[idx+1,2]*np.exp(-coefs[idx+1,2]*x_ns)
                        elif type_fit[idx+1] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx+1,0:3]))
                    elif idx == M-1:
                        if type_fit[idx-1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx-2]
                            y_inter_deriv[n] = -coefs[idx-1,1]*coefs[idx-1,2]*np.exp(-coefs[idx-1,2]*x_ns)
                        elif type_fit[idx-1] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx-1,0:3]))
                    else:
                        if type_fit[idx] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx-1]
                            y_inter_deriv[n] = -coefs[idx,1]*coefs[idx,2]*np.exp(-coefs[idx,2]*x_ns)
                        elif type_fit[idx] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx,0:3]))
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if idx_before == 0:
                        if type_fit[idx_before+1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx_before]
                            y_inter[n], y_inter_deriv[n] = coefs[idx_after,0] + coefs[idx_after,1]*np.exp(-coefs[idx_after,2]*x_ns), -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_ns)
                            y_inter_deriv[n] = -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_ns)
                        elif type_fit[idx_before+1] == "hermite":
                            y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_after,0:3]), hermval(x_inter[n],hermder(coefs[idx_after,0:3]))
                    elif idx_after == M-1:
                        if type_fit[idx_after-1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx_after-2]
                            y_inter[n], y_inter_deriv[n] = coefs[idx_after-1,0] + coefs[idx_after-1,1]*np.exp(-coefs[idx_after-1,2]*x_ns), -coefs[idx_after-1,1]*coefs[idx_after-1,2]*np.exp(-coefs[idx_after-1,2]*x_ns)
                        elif type_fit[idx_after-1] == "hermite":
                            y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_after-1,0:3]), hermval(x_inter[n],hermder(coefs[idx_after-1,0:3]))
                    else:
                        if interpolation_options["transition"] == "smooth":
                            w_left = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                            if type_fit[idx_before] == "nelson_siegel":
                                x_left = x_inter[n] - x[idx_before-1]
                                y_left, y_left_deriv = coefs[idx_before,0] + coefs[idx_before,1]*np.exp(-coefs[idx_before,2]*x_left), -coefs[idx_before,1]*coefs[idx_before,2]*np.exp(-coefs[idx_before,2]*x_left)
                            elif type_fit[idx_before] == "hermite":
                                y_left, y_left_deriv = hermval(x_inter[n],coefs[idx_before,0:3]), hermval(x_inter[n],hermder(coefs[idx_before,0:3]))
                            if type_fit[idx_after] == "nelson_siegel":
                                x_right = x_inter[n] - x[idx_after-1]
                                y_right, y_right_deriv = coefs[idx_after,0] + coefs[idx_after,1]*np.exp(-coefs[idx_after,2]*x_right), -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_right)
                            elif type_fit[idx_after] == "hermite":
                                y_right, y_right_deriv = hermval(x_inter[n],coefs[idx_after,0:3]), hermval(x_inter[n],hermder(coefs[idx_after,0:3]))
                            y_inter[n], y_inter_deriv[n] = w_left*y_left + (1-w_left)*y_right, w_left*y_left_deriv + (1-w_left)*y_right_deriv
                        else:
                            if type_fit[idx_before] == "nelson_siegel":
                                x_ns = x_inter[n] - x[idx_before-1]
                                y_inter[n], y_inter_deriv[n] = coefs[idx_before,0] + coefs[idx_before,1]*np.exp(-coefs[idx_before,2]*x_ns), -coefs[idx_before,1]*coefs[idx_before,2]*np.exp(-coefs[idx_before,2]*x_ns)
                            elif type_fit[idx_before] == "hermite":
                                y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_before,0:3]), hermval(x_inter[n],hermder(coefs[idx_before,0:3]))
    return y_inter, y_inter_deriv

def nelson_siegel_coef(x1,x2,y0,y1,y2):
    alpha = (y0-y2)/(y0-y1)
    b_hat = 2*(alpha*x1-x2)/(alpha*x1**2-x2**2)
    result = minimize(nelson_siegel_coef_obj,b_hat,method = "nelder-mead",args = (alpha,x1,x2),options={'xatol': 1e-12,"disp": False})
    if type(result.x) == np.ndarray:
        b = result.x[0]
    elif type(result.x) == int or type(result.x) == int or type(result.x) == np.int32 or type(result.x) == np.int64 or type(result.x) == np.float64:
        b = result.x
    a = (y0-y1)/(1-np.exp(-b*x1))
    f_inf = y0 - a
    return f_inf, a, b

def nelson_siegel_coef_obj(b,alpha,x1,x2):
    se = (alpha-(1-np.exp(-b*x2))/(1-np.exp(-b*x1)))**2
    return se

def swap_indices(data,T):
    for item in data:
        if item["instrument"] == "swap":
            indices = []
            if item["fixed_freq"] == "quarterly":
                for i in range(1,4*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.25,T)[1])
            elif item["fixed_freq"] == "semiannual":
                for i in range(1,2*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.5,T)[1])
            elif item["fixed_freq"] == "annual":
                for i in range(1,item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i,T)[1])
            item["indices"] = indices
    return data

def R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data):
    R_fra.reverse(), T_fra.reverse()
    R_endo = [None]*len(T_endo)
    for i in range(0,len(T_fra)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_fra[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_fra[i]*T_fra[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_fra.reverse(), T_fra.reverse()
    R_endo.reverse(), T_endo.reverse()
    for i in range(0,len(T_endo)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_endo[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_endo[i]*T_endo[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_endo.reverse(), T_endo.reverse()
    return R_endo

def zcb_curve_fra_fit_obj(R_fra,T_fra,T_known,T_endo,T_all,R_all,fra_data,interpolation_options,scaling = 1):
    sse = 0
    R_fra = list(R_fra)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra)):
        if T_fra[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_fra[i],T_all)[1]] - R_fra[i])**2
    for i in range(0,len(T_endo)):
        if T_endo[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_endo[i],T_all)[1]] - R_endo[i])**2
    sse *= scaling
    return sse

def zcb_curve_swap_fit_obj(R_knot,T_known,T_knot,T_all,R_known,swap_data,interpolation_options,scaling = 1):
    sse = 0
    R_knot = list(R_knot)
    R_all, R_deriv = interpolate(T_all,T_known + T_knot,R_known + R_knot,interpolation_options)
    p = zcb_prices_from_spot_rates(T_all,R_all)
    for n, swap in enumerate(swap_data):
        if swap["fixed_freq"] == "quarterly":
            alpha = 0.25
        if swap["fixed_freq"] == "semiannual":
            alpha = 0.5
        if swap["fixed_freq"] == "annual":
            alpha = 1
        S_swap = 0
        for idx in swap["indices"]:
            S_swap += alpha*p[idx]
        R_swap = (1 - p[swap["indices"][-1]])/S_swap
        sse += (R_swap - swap["rate"])**2
    sse *= scaling
    return sse

def value_in_list_returns_I_idx(value,list,precision = 1e-12):
    output = False, None
    for i, item in enumerate(list):
        if abs(value-item) < precision:
            output = True, i
            break
    return output

def idx_before_after_in_iterable(value,list):
    idx_before, idx_after = None, None
    if value < list[0]:
        idx_before, idx_after = None, 0
    elif list[-1] < value:
        idx_before, idx_after = len(list) - 1, None
    else:
        for i in range(0,len(list)):
            if list[i] < value:
                idx_before = i
            elif list[i] > value:
                idx_after = i
                break
    return idx_before, idx_after

def value_in_list_of_dict_returns_I_idx(value,L,name,precision = 1e-12):
    output = False, None
    for item in L:
        if abs(value-item[name]) < precision:
            output = True, item
            break
    return output
