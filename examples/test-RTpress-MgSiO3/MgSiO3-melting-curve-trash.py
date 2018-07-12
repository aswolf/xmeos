def trace_inconsistent_hybrid_phase_bound(Pinit, Tinit, Plim, 
                             sol_eos, liq_thermal_eos, liq_compress_eos,
                             Vinit_sol=None, Vinit_liq_compress=None, N=100):
    
    Pbound = np.linspace(Pinit, Plim, N)
    Tbound = np.zeros(N)
    Tbound[0] = Tinit
    
    dP = Pbound[1]-Pbound[0]
    
    Vsol_init = volume(Pinit, Tinit, sol_eos, Vinit=Vinit_sol)
    Vliq_init = volume(Pinit, Tinit, liq_compress_eos, Vinit=Vinit_liq_compress)

    for ind, (P, T) in enumerate(zip(Pbound[:-1], Tbound[:-1])):
        Vsol = volume(P, T, sol_eos, Vinit=Vsol_init)
        Vliq = volume(P, T, liq_compress_eos, Vinit=Vliq_init)
        
        dV = Vliq-Vsol
        dS = (liq_thermal_eos.entropy(Vliq, T) 
              - sol_eos.entropy(Vsol, T))[0]
        
        dTdP = dV/dS/CONSTS['PV_ratio']
        dT = dTdP*dP
        
        Pbound[ind+1] = P+dP
        Tbound[ind+1] = T+dT
        
        Vsol_init = Vsol
        Vliq_init = Vliq

    return Pbound, Tbound

def hybrid_gamma(T, V, liq_compress_eos, liq_thermal_eos):
    dPdT = (liq_compress_eos.press(V, T+.5)
            - liq_compress_eos.press(V, T-.5))
    Cv = liq_thermal_eos.heat_capacity(V, T)
    gamma = V*dPdT/Cv/CONSTS['PV_ratio']
    return gamma

def dTdV_ad(T, V, liq_compress_eos, liq_thermal_eos):
    gamma = hybrid_gamma(T, V, liq_compress_eos, liq_thermal_eos)
    dTdV = -gamma*T/V
    return dTdV


def hybrid_volume(P, T, liq_compress_eos=liq_compress_eos, Vinit=9):
    V = volume(P, T, liq_compress_eos, Vinit=Vinit)
    return V

def hybrid_entropy(V, T, dS_fus=1.5, P_fus0=25, T_fus0=2900,
                   sol_eos=mgpv_eos,
                   liq_compress_eos=liq_compress_eos,
                   liq_thermal_eos=liq_thermal_eos):
    
    Vsol_fus0 = volume(P_fus0, T_fus0, sol_eos)
    S_sol0 = sol_eos.entropy(Vsol_fus0, T_fus0)
    
    V_fus0 = hybrid_volume(P_fus0, T_fus0, liq_compress_eos)
    V_path = np.hstack((V_fus0,V))
    Tad = sp.integrate.odeint(
        dTdV_ad, T_fus0, V_path , 
        args=(liq_compress_eos, liq_thermal_eos))[1:,0]

    dS = (liq_thermal_eos.entropy(V, T) 
          -liq_thermal_eos.entropy(V, Tad))
    
    S = S_sol0 + dS_fus*CONSTS['kboltz'] + dS
    return S