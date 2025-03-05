import numpy as np
from scipy.integrate import solve_ivp


class catalysis_network:
  def __init__(self,
            s_mat,
            xcat_sym=np.array([]),
            vcat_sym=np.array([])):
    """Initiate a catalysis_network object

    Parameters
    ----------
    s_mat : numpy array, n_cat-by-m_cat
      The stoichiometry matrix defining the catalysis network.
      Not necessarily full rank.
      n_cat is the number of rows, i.e. the number of variables for catalysis
        dynamics, same as length of xcat
      m_cat is the number of columns, i.e. the number of catalysis reactions,
        same as length of kcat.
    xcat_sym : numpy array of sympy symbols, optional
      Symbols for xcat variables. Length n_cat.
    vcat_sym : numpy array of sympy symbols, optional
      Symbols for vcat variables, fluxes of catalysis reactions. Length m_cat.
    """
    self.s_mat=s_mat
    self.dim_ncat,self.dim_mcat=s_mat.shape

class binding_and_catalysis:
  def __init__(self,bn,cn,kbind,kcat,total_const,total_const_idx,xcat_in_total_idx,cat_active_in_xbind_idx):
    """Initiate a binding_and_catalysis object, which contains a binding network
    and a catalysis network, together with links between them.
    On the catalysis time scale, the binding constants kbind don't change,
        and the

    Parameters
    ----------
    bn : a binding_network object
      The binding network that specifies how the catalysis fluxes are regulated.
        It defines the map from xcat (catalysis variables) to concentrations
          of catalytic active species responsible for the catalysis fluxes.
        Flux v = kcat * cat_active
      It has to contain all the catalytic active species of the catalysis reactions,
        even if there are reactions that has constant rates. (That correspond to
        a binding reaction that is just one atomic entry with constant total.)

    cn : a catalysis_network object
      The catalysis network that specifies how the fluxes change molecular concentrations.


    xcat_in_total_idx: a tuple of integers, length <= self.dim_ncat
      indices for xcat as totals in the binding network bn.
      Length could be less than self.dim_ncat, since not necessarily all xcat
        are contained in totals.
      The xcat contained in totals NEED TO COME FIRST in the ordering of xcat.
      xcat=np.concatenate((total[xcat_in_total_idx], xcat[]]

    cat_active_in_xbind_idx: a tuple of integers, length self.dim_mcat
      Indices that the active species for the catalysis reactions correspond to
        in the xbind species in the binding network bn.
    """
    self.bn=bn
    self.kbind=kbind # binding network's binding constants are fixed at the catalysis timescale
    self.kcat=kcat
    self.logkbind=np.log10(kbind)
    self.total_const=total_const
    self.logtotal_const=np.log10(total_const)
    self.total_const_idx=total_const_idx
    self.cn=cn
    self.xcat_in_total_idx=xcat_in_total_idx
    self.cat_active_in_xbind_idx=cat_active_in_xbind_idx

    # # Extend cn.s_mat to shape bn.dim_d by cn.dim_mcat so that the dynamics for
    # #   every total in the binding networks is contained.
    # # These extra totals don't change, so their rows of s_mat are simply zeros.
    # self.s_mat_total=np.zeros((bn.dim_d,cn.dim_mcat))
    # self.s_mat_total[cn.xcat_in_total_idx,:]=cn.s_mat
    # MAYBE, no need to extend. Since all totals that are changed in the catalysis
    #   time scale are already contained in xcat. If they are not contained, then
    #   they don't change, so they can be plugged in as just a parameter.

  def dlogxcatdt(self,logxcat,a_mat):
    """Calculate dlogxcat/dt for at a given logxcat point.

    Parameters
    ----------
    logxcat : numpy array, length dim_ncat
      The vector of xcat in log10 indicating the current state of catalysis
        dynamics.
      Entries of xcat corresponding to totals in the binding network need to
        come first.
    a_mat : numpy array, bn.dim_d by bn.dim_n
      Matrix defining totals in terms of xbind in the binding network.
      Correspond to conserved quantities at the binding timescale.
      The default choice should be bn.l_mat.
      Can be modified from bn.l_mat to describe restrictions or asymptotic
        limits of the system.
    """
    cn=self.cn
    bn=self.bn
    logtk=np.zeros(bn.dim_n)
    n_contained=len(self.xcat_in_total_idx)
    logtk[self.xcat_in_total_idx]=logxcat[:n_contained] # totals come first in logtk
    #   and only n_contained number of xcat are contained in totals.
    # Then the last r are kbin.
    logtk[self.total_const_idx]=self.logtotal_const
    # IS IT TRUE that all totals are contained in xcat? No. Some are constant...
    logtk[bn.dim_d:]=self.logkbind
    logxbind=bn.tk2x_num(logtk, a_mat=a_mat)
    # print('logxbind', logxbind)
    vcat=self.kcat.T @ 10**logxbind[self.cat_active_in_xbind_idx]  # 2024/08/08 added transposition of kcat
    vcat = np.expand_dims(vcat, axis=0)  # 2024/08/08 added
    new_logxcat = (cn.s_mat@vcat) / 10**logxcat / np.log(10) # divide by xcat so that this kinetics of logxcat.
    return new_logxcat
  
  def get_traj(self,logxcat_init,t_init,t_end,npts,a_mat=np.array([]),get_logder_xbind=True):
    if not np.any(a_mat): #if a_mat is not specified, use l_mat by default.
      a_mat=self.bn.l_mat
    func_int=lambda t,y:self.dlogxcatdt(y,a_mat)
    y0=logxcat_init
    t_span=(t_init,t_end)
    t_eval=np.linspace(t_init,t_end,npts)

    def events(t, y):
      return y[0] > -10  # S in solution less than 1e-8 is not considable.
    
    events.terminal = True

    sol=solve_ivp(func_int,t_span,y0,t_eval=t_eval, events=events, method='BDF')  # BDF more suitalbe for stiff ODE
    logxcat_traj=sol.y.T # transpose so time points come in the first dim.

    if get_logder_xbind:
      logder_traj,logxbind_traj=self.get_logder_xbind_from_traj(logxcat_traj,a_mat=a_mat)
      return sol.t,logxcat_traj,logxbind_traj,logder_traj
    else:
      return sol.t,logxcat_traj

  def get_logder_xbind_from_traj(self,logxcat_traj,a_mat=np.array([])):
    if not np.any(a_mat): #if a_mat is not specified, use l_mat by default.
      a_mat=self.bn.l_mat
    cn=self.cn
    bn=self.bn
    npts=logxcat_traj.shape[0]
    logtk_traj=np.zeros((npts,bn.dim_n))
    n_contained=len(self.xcat_in_total_idx)
    logtk_traj[:,self.xcat_in_total_idx]=logxcat_traj[:,:n_contained] # totals come first in logtk
    logtk_traj[:,self.total_const_idx]=self.logtotal_const
    logtk_traj[:,bn.dim_d:]=self.logkbind
    logder_traj,logxbind_traj=bn.logder_num(logtk_traj,chart='tk')
    return logder_traj,logxbind_traj
  