# Author: Fangzhou Xiao, 20230720

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.stats import dirichlet
from scipy.spatial import HalfspaceIntersection, ConvexHull, Delaunay
from scipy.optimize import linprog
import itertools
import cvxpy as cp

class rop_ld_regime:
  """
  The ld_regime object, for a given ld (log derivative or reaction order),
    there could be multiple dominance regimes (defined by dominance relations in x
    for a given catalytic activity on top of a binding network).
  Each ld_regime has possibly several dominance regimes with the same ld.

  Parameters
  ----------
  ld : a tuple of integers
    the log derivative, or reaction order, of this ld_regime.
  b_vec : a numpy array vector
    the b vector defining the activity, b^T x.
  bn : a binding_network object
    the binding network that this ld_regime belongs to.
  is_ray : boolean
    whether this ld_regime is a ray or not.
  dom_regime_keys : list of tuples
    The list of tuples, each tuple is the key for a dominance regime that has
    reaction order (ld) the same as this ld_regime.
  dom_regime_dict : dictionary
    The dictionary with key as the tuple (perm,dom_idx) representing a dominance regime
     and value of the dominance regime object.
  neighbors_dict : dictionary
    The dictionary for ld_regime neighbors of this ld_regime.
    Has four keys, three of them are "finite", "infinite", and "all",
      for finite, infinite (ray) ld_regime neighbors, and all of ld_regime
      neighbors, respectively.
    For each of these keys, we get a dictionary as well, with ld tuple as key and
      the ld_regime object as value.
    The last key is "zero", which maps to neighbors connected via ld_regimes
      that are zero rays. The value is itself a dictionary with "finite", "infinite",
      and "all" mapping to dictionaries with (ld:ld_regime) entries.
  is_feasible : boolean
    Whether this ld_regime is feasible. It is feasible if it has at least one dom_regime
    that is feasible.
  neighbors_constrained_dict : dictionary
    The dictionaory for ld_regime neighbors of this ld_regime that are feasible
    under constraints applied to each dom_regime.
  """
  # b_vec defines an activity, yielding possibly multiple regimes for each vertex,
  # with logder corresponding to different rows of the vertex logder matrix.
  # ld_regime focus on logder, collapsing the same logder to be the same ld_regime,
  # which may come from different vertices.
  # Each ld_regime has multiple dom_regimes and regions of feasibility.
  def __init__(self,ld,is_ray,b_vec,dom_regime_keys,bn):
    '''
    Parameters
    ----------
    ld : a tuple of integers
      the log derivative, or reaction order, of this ld_regime.
    is_ray : boolean
      whether this ld_regime is a ray or not.
    b_vec : a numpy array vector
      the b vector defining the activity, b^T x.
    dom_regime_keys : list of tuples
      The list of tuples, each tuple is the key for a dominance regime that has
      reaction order (ld) the same as this ld_regime.
    bn : a binding_network object
      the binding network that this ld_regime belongs to.
    '''
    self.ld=ld
    self.b_vec=tuple(b_vec)
    self.bn=bn
    self.is_ray=is_ray
    self.dom_regime_keys=dom_regime_keys
    self.dom_regime_dict={key:bn.activity_regime_dict[self.b_vec]['all'][key] for key in dom_regime_keys}
    self.is_feasible=True

  def find_neighbors(self):
    # Construct self.neighbors_dict based on dominance regimes' neighbors,
    # then construct neighbors_constrained_dict.
    neighbors_fin={};neighbors_inf={};neighbors_fin_zero={};neighbors_inf_zero={}
    # Instead of iterating through all ld_regimes, we just look at this ld_regime's
    # dom_regimes and their corresponding ld_regimes as neighbors.
    for key,dom_regime in self.dom_regime_dict.items():
      for nb_key,nb_regime in dom_regime.neighbors_dict['finite'].items():
        neighbors_fin[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['finite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_dict['infinite'].items():
        neighbors_inf[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['infinite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_dict['zero']['finite'].items():
        neighbors_fin_zero[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['finite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_dict['zero']['infinite'].items():
        neighbors_inf_zero[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['infinite'][nb_regime.ld]
    neighbors_all={**neighbors_fin,**neighbors_inf}
    neighbors_all_zero={**neighbors_fin_zero,**neighbors_inf_zero}
    self.neighbors_dict={'all':neighbors_all,'finite':neighbors_fin,'infinite':neighbors_inf,
                         'zero':{'all':neighbors_all_zero,'finite':neighbors_fin_zero,'infinite':neighbors_inf_zero}}

  def update_feasibility(self):
    # Construct self.dom_regime_constrained_dict using dom_regime's is_feasible tag.
    # Then update self.is_feasible tag, True if self.dom_regime_constrained_dict is not empty,
    # since this means there are some dom_regimes satisfying the constraints.
    self.dom_regime_constrained_dict={key:regime for key,regime in self.dom_regime_dict.items() if regime.is_feasible}
    if self.dom_regime_constrained_dict: self.is_feasible=True
    else: self.is_feasible=False

  def update_constrained_neighbors(self):
    # Construct self.neighbors_constrained_dict based on dominance regimes' neighbors_constrained_dict.
    # This does not distinguish neighbors via zero or not.

    neighbors_constrained_fin={}
    neighbors_constrained_inf={}
    # Instead of iterating through all ld_regimes, we just look at this ld_regime's
    # dom_regimes and their corresponding ld_regimes as neighbors.
    for key,dom_regime in self.dom_regime_constrained_dict.items():
      for nb_key,nb_regime in dom_regime.neighbors_constrained_dict['finite'].items():
        neighbors_constrained_fin[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['finite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_constrained_dict['infinite'].items():
        neighbors_constrained_inf[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['infinite'][nb_regime.ld]
    neighbors_constrained_all={**neighbors_constrained_fin,**neighbors_constrained_inf}
    self.neighbors_constrained_dict={'all':neighbors_constrained_all,
                                     'finite':neighbors_constrained_fin,
                                     'infinite':neighbors_constrained_inf}

class rop_dom_regime:
  """
  A dominance regime (dom_regime) object for a catalytic activity on top of
    a given binding network defined by b^T x, for some b vector.
  A dominance regime is labeled by a tuple ((perm),j) where perm is the permutation
    (a length-n tuple) defining the vertex, and j is the dominant species index
    in activity b^T x.

  Parameters
  ----------
  row_idx : integer
    the integer j indicating x_j is the dominant species in b^T x at this regime.
  b_vec : numpy array vector
    The b vector defining the activity b^T x.
  vertex_perm : tuple of integers
    The tuple of length d (number of totals of binding network) refering to the
      dominance vertex that this dom_regime belongs to.
  vertex : a ROP_vertex object
    The vertex that this dom_regime belongs to. This dom_regime is at this vertex
      with an additional dominance condition for the activity.
  bn : a binding_network object
    the binding network that this dom_regime belongs to.
  ld : tuple
    The reaction order or log derivative of this dom_regime.
    The row_idx row of the vertex's h_mat.
  neighbors_dict : dictionary
    The dictionary for dom_regime neighbors of this dom_regime.
    Has four keys, three of them are "finite", "infinite", and "all",
      for finite, infinite (ray) dom_regime neighbors, and all of dom_regime
      neighbors, respectively.
    For each of these keys, we get a dictionary as well, with a (perm,row_idx)
      tuple as key and the dom_regime object as value.
    The last key is "zero", which maps to neighbors connected via dom_regimes
      that are zero rays in reaction orders (or log derivative).
      The value is itself a dictionary with "finite", "infinite",
      and "all" mapping to dictionaries with (ld:ld_regime) entries.
  is_feasible : bool
    Whether this dom_regime is feasible under the last given constraints.
    If a new set of constraints is given and tested, then this is overwritten.
  neighbors_constrained_dict : dictionary
    The dictionaory for ld_regime neighbors of this ld_regime that are feasible
      under constraints applied to each dom_regime.
    Same keys as neighbors_dict.
  
  c_mat_add_x : numpy array
    Additional inequalities specifying the condition to reach this 
      dom_regime on top of the inequalities from its vertex.
    To be concatenated vertically with vertex.c_mat_x.
    This is in chart 'x'.
  c_mat_add_xak : numpy array
    Additional inequalities specifying the condition to reach this 
      dom_regime on top of the inequalities from its vertex.
    To be concatenated vertically with vertex.c_mat_xak.
    This is in chart 'xak'.
  c0_vec_add : numpy array
    Additional intercept vector entries for the inequalities specifying
      the condition to reach this dom_regime on top of the inequalities
      from its vertex.
    To be concatenated vertically with vertex.c0_vec when used to
      specify inequalities.
    Same vector for both chart 'x' and 'xak'.
  c_mat_add_tk : numpy array
    Additional inequalities specifying this dom_regime on top of the 
      inequalities from its vertex.
    To be concatenated vertically with vertex.c_mat_tk.
    This is in chart 'tk'.
    Is only meaningful if its vertex is non-singular.
  """
  def __init__(self,row_idx,b_vec,vertex_perm,bn):
    """Initiates a ROP vertex.

    Parameters
    ----------
    row_idx : integer
      the integer j indicating x_j is the dominant species in b^T x at this regime.
    b_vec : numpy array vector
      The b vector defining the activity b^T x.
    vertex_perm : tuple of integers
      The tuple of length d (number of totals of binding network) refering to the
      dominance vertex that this dom_regime belongs to.
    bn : a binding_network object
      the binding network that this dom_regime belongs to.
    """
    self.row_idx=row_idx
    self.b_vec=tuple(b_vec)
    self.vertex_perm=vertex_perm
    self.vertex=bn.vertex_dict['all'][vertex_perm]
    self.bn=bn
    self.ld=tuple(self.vertex.h_mat[row_idx,:])
    self.is_feasible=True

  def chart_check_add(self,chart):
    if chart=='x':
      try: 
        c_mat_add=self.c_mat_add_x
        c0_vec_add=self.c0_vec_add
      except AttributeError: # c_mat_add is not yet calculated
        self.calc_c_mat_add_x()
        c_mat_add=self.c_mat_add_x
        c0_vec_add=self.c0_vec_add
    elif chart=='xak':
      try: 
        c_mat_add=self.c_mat_add_xak
        c0_vec_add=self.c0_vec_add
      except AttributeError: # c_mat_add_xak is not yet calculated
        self.calc_c_mat_add_xak()
        c_mat_add=self.c_mat_add_xak
        c0_vec_add=self.c0_vec_add
    elif chart=='tk':
      try: 
        c_mat_add=self.c_mat_add_tk
        c0_vec_add=self.c0_vec_add_tk
      except AttributeError: # c_mat_add_tk is not yet calculated
        self.calc_c_mat_add_tk()
        c_mat_add=self.c_mat_add_tk
        c0_vec_add=self.c0_vec_add_tk
    else:
      raise Exception('chart that is not one of "x,xak,tk" is not implemented yet')
    return c_mat_add,c0_vec_add

  def feasibility_test(self,chart='x',opt_constraints=[],positive_threshold=1e-5,is_asymptotic=True):
    c_mat_add,c0_vec_add = self.chart_check_add(chart)
    vv=self.vertex
    opt_var=self.bn.opt_var
    opt_constraints_test=[]
    opt_constraints_test+=opt_constraints
    if np.any(c_mat_add): #c_mat_add could be empty if b_vec has just one nonzero entry.
      if is_asymptotic: # if the test condition is asymptotic, c0_vec_add is considered 0.
        opt_constraints_test+=[c_mat_add @ opt_var >= positive_threshold]
      else: # if not asymptotic, it is exact, so c0_vec_add needs to be considered.
        opt_constraints_test+=[c_mat_add @ opt_var + self.c0_vec_add >= positive_threshold]
    is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints_test,is_asymptotic=is_asymptotic)
    return is_feasible

  def calc_c_mat_add_x(self):
    # Compute feasibility conditions of this dominance regime in addition 
    #   to vertex feasibility conditions in the form of 
    #   c_mat_add_x * logx + c0_vec_add > 0, i.e. in chart x.
    # Each condition comes from and inequality of the form 
    #   b_j1 x_j1 > b_j2 x_j2, which can be written as 
    #   log(x_j1) - log(x_j2) + [log(b_j1) - log(b_j2)] > 0.
    # So the corresponding row of c_mat_add_x are all 0's except 1 at j1 
    #   and -1 at j2. And corresponding entry of c0_vec is log(b_j1) - log(b_j2).
    j=self.row_idx
    b_vec=np.array(self.b_vec)
    idx_nonzero_b=np.where(b_vec > 0)[0]
    n_ineq=len(idx_nonzero_b)-1
    c_mat_add_x=np.zeros((n_ineq,self.bn.dim_n))
    c0_vec_add=np.zeros(n_ineq)
    counter=0
    for jp in idx_nonzero_b:
      if jp!=j:
        c_mat_add_x[counter,j]=1
        c_mat_add_x[counter,jp]=-1
        c0_vec_add[counter]=np.log10(b_vec[j])-np.log10(b_vec[jp])
        counter+=1
    self.c_mat_add_x=c_mat_add_x
    self.c0_vec_add=c0_vec_add

  def calc_c_mat_add_xak(self):
    # compute feasibility conditions of this dominance regime in addition to vertex feasibility conditions
    # in the form of c_mat_add_xak * log(xa,k) + c0_vec_add > 0, i.e. in chart log(xa,k)
    try: c_mat_add_x=self.c_mat_add_x
    except AttributeError: # c_mat_add is not yet calculated
      self.calc_c_mat_add_x()
      c_mat_add_x=self.c_mat_add_x
    try: xak2x_map=self.bn.xak2x_map
    except AttributeError:
      self.bn.calc_xak2x_map()
      xak2x_map=self.bn.xak2x_map
    c_mat_add_xak=c_mat_add_x.dot(xak2x_map)
    self.c_mat_add_xak=c_mat_add_xak

  def calc_c_mat_add_tk(self):
    # compute feasibility conditions of this dominance regime in addition to vertex feasibility conditions
    # in the form of c_mat_add_tk * log(t,k) > 0, i.e. in chart log(t,k)
    assert self.vertex.orientation!=0, "only finite vertices can have non-singular (t,k) chart"
    try: c_mat_add_x=self.c_mat_add_x
    except AttributeError: # c_mat_add is not yet calculated
      self.calc_c_mat_add_x()
      c_mat_add_x=self.c_mat_add_x
    # The equivalence here is the following:
    # c_mat_tk @ log(t,k) + c0_vec_tk >=0  <=>  c_mat_x @ logx + c0_vec >=0
    # and we use h_mat*log(t,k)=log(x) at the vertex
    c_mat_add_tk=c_mat_add_x.dot(self.vertex.h_mat) 
    c0_vec_add_tk=self.c0_vec_add - c_mat_add_x.dot(self.h_mat.dot(self.m0_vec))
    self.c_mat_add_tk=c_mat_add_tk
    self.c0_vec_tk=c0_vec_add_tk

  def find_neighbors(self):
    vv=self.vertex
    neighbors_fin={}
    neighbors_inf={}
    regime_fin_dict=self.bn.activity_regime_dict[tuple(self.b_vec)]['finite']
    regime_inf_dict=self.bn.activity_regime_dict[tuple(self.b_vec)]['infinite']
    for key,regime in regime_fin_dict.items():
      perm,row_idx=key
      # if same vertex, but different dominance, then it's neighbor.
      if perm==self.vertex_perm and row_idx!=self.row_idx:
        neighbors_fin[key] = regime
      # if different vertex, but it's a neighboring vertex, then it's neighbor.
      else:
        try: # if perm is in neighbors
          vv.neighbors_dict['all'][perm]
          # If the neighboring vertex's transition index (the row_idx that changed compared to self)
          # is the same as its dominance index, then it's a neighbor.
          # Or, if the transition index is not the same, then its dominance index needs to be the same as self.
          if row_idx == self.row_idx or perm[np.where(np.array(perm)!=self.vertex_perm)[0][0]]==row_idx:
            neighbors_fin[key] = regime
        except KeyError: # not in neighbors
          continue


    for key,regime in regime_inf_dict.items():
      perm,row_idx=key
      # if same vertex, but different dominance, then it's neighbor.
      if perm==self.vertex_perm and row_idx!=self.row_idx:
        neighbors_inf[key] = regime
      # if different vertex, but it's a neighboring vertex, then it's neighbor.
      else:
        try: # if perm is in neighbors
          vv.neighbors_dict['all'][perm]
          if row_idx == self.row_idx or perm[np.where(np.array(perm)!=self.vertex_perm)[0][0]]==row_idx:
            neighbors_inf[key] = regime
        except KeyError: # not in neighbors
          continue
    neighbors_all = {**neighbors_fin,**neighbors_inf}
    self.neighbors_dict={'finite':neighbors_fin,'infinite':neighbors_inf,'all':neighbors_all}

  def find_neighbors_zero(self):
    # Find neighbors that connect through dom_regimes with zero ld.
    # neighbors via zero are not considered neighbors of dom_regimes,
    # but we would like to keep them tracked, since they will become neighbors
    # in ld space.
    # This should be done after all neighbors of dom_regimes are constructed
    # since it relies on knowing the neighbors of dom_regime with ld=0.
    # A zero neighbor could have a zero neighbor, so recursion is needed to exhaust this.
    def extract_zero_neighbor(infinite_neighbors, finite_zero_neighbors, infinite_zero_neighbors, ld_zero, visited):
      zero_neighbors = {
          key: dom_regime for key, dom_regime in infinite_neighbors.items() if tuple(dom_regime.ld) == ld_zero
      }
      if zero_neighbors:  # the dictionary is not empty
          for key, dom_regime in zero_neighbors.items():
              if key not in visited:
                  visited.add(key)
                  finite_zero_neighbors.update(dom_regime.neighbors_dict['finite'])
                  infinite_zero_neighbors.update(dom_regime.neighbors_dict['infinite'])
                  finite_zero_neighbors, infinite_zero_neighbors, visited = extract_zero_neighbor(
                      dom_regime.neighbors_dict['infinite'],
                      finite_zero_neighbors,
                      infinite_zero_neighbors,
                      ld_zero,
                      visited
                  )
      return finite_zero_neighbors, infinite_zero_neighbors, visited

    n = self.bn.dim_n
    ld_zero = tuple(np.zeros(n))
    finite_zero_neighbors = {}
    infinite_zero_neighbors = {}
    visited = set()

    infinite_neighbors = self.neighbors_dict['infinite']
    finite_zero_neighbors, infinite_zero_neighbors, visited = extract_zero_neighbor(
        infinite_neighbors, finite_zero_neighbors, infinite_zero_neighbors, ld_zero, visited
    )

    # Remove self from the neighbors' dictionaries if present
    try:
        if self.vertex.orientation == 0:
            del infinite_zero_neighbors[(self.vertex_perm, self.row_idx)]
        else:
            del finite_zero_neighbors[(self.vertex_perm, self.row_idx)]
    except KeyError:
        pass

    all_zero_neighbors = {**finite_zero_neighbors, **infinite_zero_neighbors}
    self.neighbors_dict['zero'] = {
        'finite': finite_zero_neighbors,
        'infinite': infinite_zero_neighbors,
        'all': all_zero_neighbors
    }
    self.neighbors_dict['allnzero'] = {
        'finite': {**finite_zero_neighbors,**self.neighbors_dict['finite']},
        'infinite': {**infinite_zero_neighbors,**self.neighbors_dict['infinite']},
        'all': {**all_zero_neighbors,**self.neighbors_dict['all']}
    }

  def update_constrained_neighbors(self):
    # Using dom_regimes' is_feasible tag to have updated neighbors.
    # The feasible neighbors of an infeasible neighbor becomes this vertex's neighbors.
    # We search the next level for the infeasible neighbors of an infeasible neighbor.
    # So this uses recursion.

    def extract_infeasible_neighbor(infeasible_neighbors,neighbors_feasible_all,visited):
      for key,dom_regime in infeasible_neighbors.items():
          if key not in visited:
              visited.add(key)
              # get the feasible and infeasible neighbors of this dom_regime
              infeasible_neighbors = {}
              feasible_neighbors={}
              for key,dom_regime in dom_regime.neighbors_dict['allnzero']['all'].items():
                if dom_regime.is_feasible:
                  feasible_neighbors[key]=dom_regime
                else:
                  infeasible_neighbors[key]=dom_regime
              # the feasible ones are neighbors under constraint.
              neighbors_feasible_all.update(feasible_neighbors)
              # the infeasible ones we need to look further
              if infeasible_neighbors:
                neighbors_feasible_all, visited = extract_infeasible_neighbor(
                  infeasible_neighbors,
                  neighbors_feasible_all,
                  visited
                )
      return neighbors_feasible_all, visited

    neighbors_feasible_all = {key:dom_regime for key,dom_regime in self.neighbors_dict['allnzero']['all'].items() if dom_regime.is_feasible}
    infeasible_neighbors = {key:dom_regime for key,dom_regime in self.neighbors_dict['allnzero']['all'].items() if not dom_regime.is_feasible}
    visited = set()
    if infeasible_neighbors:
      neighbors_feasible_all,visited = extract_infeasible_neighbor(
        infeasible_neighbors,neighbors_feasible_all,visited
      )
    # Remove self from the neighbors' dictionaries if present
    try:
        del neighbors_feasible_all[(self.vertex_perm,self.row_idx)]
    except KeyError:
        pass
    neighbors_feasible_inf = {
        key:dom_regime for key,dom_regime in neighbors_feasible_all.items() if dom_regime.vertex.orientation==0
        }
    neighbors_feasible_fin = {
        key:dom_regime for key,dom_regime in neighbors_feasible_all.items() if dom_regime.vertex.orientation!=0
        }
    self.neighbors_constrained_dict={'finite':neighbors_feasible_fin,
                                  'infinite':neighbors_feasible_inf,
                                  'all':neighbors_feasible_all}

  # def print_validity_condition(self,is_asymptotic=False):
  #   # print the expression for t=x, x(t,k) and inequalities for the
  #   # region of validity for this dominance regime,
  #   # using the labels of x,t,k

  def hull_sampling(self,nsample,chart='x', margin=0,logmin=-6,logmax=6,c_mat_extra=[],c0_vec_extra=[]):
    """
    Sample points in the dom_regime's region of validity based on its hull of feasible regions.
    Extra conditions (linear) can be added as c_mat_extra @ var + c0_vec_extra >= 0.
    This is done by adding dom_regime's additional constraints to its vertex's sampling function.

    Parameters
    ----------
    nsample : int
      Number of points to be sampled.
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'.
    margin : float, optional
      The dom_regime's feasibility conditions are inequalities, 
        of the form c_mat*x + c0_vec >= margin (e.g. in 'x' chart),
        where margin is the margin used here. Default to 0.
      This can be adjusted to be stronger/weaker requirements on dominance.
    logmin : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    logmax : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    c_mat_extra : ndarray, shape (n_constraints, n_var)
      Extra optimization constraints to be added to feasibility conditions,
        in the form of c_mat_extra @ var + c0_vec_extra >= 0.
    c0_vec_extra : numpy vector, shape (n_constraints,)
      Extra optimization constraints to be added to feasibility conditions,
        in the form of c_mat_extra @ var + c0_vec_extra >=0.

    Returns
    -------
    sample : ndarray of shape nsample-by-dim_n
      dim_n is number of species in the binding network.
      Sampled points satisfying the feasibility conditions of this vertex.
      Each row (sample[i,:]) is a sampled point.
    """
    # Get the additional constraints for the dom_regime
    c_mat_add,c0_vec_add=self.chart_check_add(chart=chart)
    # Combine dom_regime constraints with given additional constraints 
    #   to get the full constraints to be added to vertex validity.
    if np.any(c_mat_extra): # if there are additional constraints
      c_mat_extra_full=np.vstack((c_mat_add,c_mat_extra))
      # Incorporate margin into c0_vec_add, since this won't be added again later in the vertex.
      c0_vec_extra_full=np.concatenate((c0_vec_add-margin*np.ones(c0_vec_add.shape[0]),c0_vec_extra))
    else: # there are no additional constraints
      c_mat_extra_full = c_mat_add
      c0_vec_extra_full = c0_vec_add-margin
    # Get the hull using the vertex's method, but with additional
    #   constraints from the dom_regime.
    points,hull,_,_=self.vertex.vertex_hull_of_validity(chart=chart,margin=margin,logmin=logmin,logmax=logmax,c_mat_extra=c_mat_extra_full,c0_vec_extra=c0_vec_extra_full)
    ncoeffs=points.shape[0]
    temp=np.sort(np.random.rand(nsample,ncoeffs-1),axis=1)
    coeffs=np.diff(temp,prepend=0,append=1,axis=1)
    sample=coeffs@points # this has shape nsample-by-dim_n
    return sample


class rop_vertex:
  """
  A vertex object for reaction order polyhedra.
  Each binding network has multiple vertices.

  Parameters
  ----------
  perm : an integer tuple
    an integer tuple of length dim_d, indicating for each conserved quantity, which species is dominant.
  bn : a binding_network object
    The binding network that this vertex belongs to.
    Has l_mat (conservation law matrix) and n_mat (stoichiometry matrix).
  p_mat : numpy array, d-by-n
    d x n matrix with exactly one nonzero entry in each row, of value 1. One-hot representation of perm.
  p0_vec : numpy array, d-by-1
    Vector used in (logt, logk) = [p_mat n_mat]' logx + [p0_vec 0]'. 
    Intercept relating log x  to (logt, logk).
    n_mat is that of the binding network.
    p0_vec is all zeros if all entries of bn's l_mat are 0 and 1's.
  m_mat : numpy array, n-by-n
    Matrix formed by p_mat and n_mat stacked vertically. 
    In other words, m_mat = [p_mat n_mat]'.
    n_mat is that of the binding network.
  m0_vec : numpy array, n-by-1.
    Vector used in (logt, logk) = m_mat logx + m0_vec.
    m0_vec is the same as [p0_vec 0]', i.e. p0_vec vertically 
      extended with r more zeros.
  orientation : an integer
    take value in +1,0,-1. Sign of determinant of [A' N'] matrix.
  h_mat : numpy array, n-by-n
    n x n matrix corresponding to the log derivative of this vertex.
    If finite (non-singular), then this is the log derivative
    If infinite (singular), then this is the direction that log derivative goes into.
    At this vertex, we have relation logx = h_mat (logt, logk) + h0_vec.
    Not always defined, computed and stored once log derivative is 
      computed by calling self.vertex_ld_calc().
  h0_vec : numpy array, n-by-1
    The intercept vector used in the following relation at this vertex:
      logx = h_mat (logt, logk) + h0_vec.
    Only defined for finite (non-singular) vertices.
    Not always defined, computed and stored once log derivative is 
      computed by calling self.vertex_ld_calc().
  c_mat_x : numpy array, shape (n_constraints, n_var)
    Matrix encoding feasibility condition of this vertex in 'x' chart, c_mat_x * logx + c0_vec > 0.
    If the feasibility condition is considered "asymptotic", i.e. in positive
      projective measure rather than Lebesgue measure (so a ray is an infinitesimal
      of volume, not a point), then c0_vec is dropped.
    Not always defined, computed and stored when used in feasibility tests.
  c_mat_xak : numpy array, shape (n_constraints, n_var)
    Matrix encoding feasibility condition of this vertex in 'xak' chart, c_mat_xak * (logxa, logk) + c0_vec > 0.
    If the feasibility condition is considered "asymptotic", i.e. in positive
      projective measure rather than Lebesgue measure (so a ray is an infinitesimal
      of volume, not a point), then c0_vec is dropped.
    Not always defined, computed and stored when used in feasibility tests.
  c0_vec : numpy vector, shape (n_constraints,)
    Numpy vector encoding a part of the feasibility condition of this vertex, same
      for 'x' chart and 'xak' chart.
    Not always defined, computed and stored when used in feasibility tests.
  c_mat_tk : numpy array, shape (n_constraints, n_var)
    Matrix encoding feasibility condition of this vertex in 'tk' chart, c_mat_tk * (logt, logk) + c0_vec_tk > 0.
    If the feasibility condition is considered "asymptotic", i.e. in positive
      projective measure rather than Lebesgue measure (so a ray is an infinitesimal
      of volume, not a point), then c0_vec_tk is dropped.
    Only defined for finite (non-singular) vertices.
    Not always defined, computed and stored when used in feasibility tests.
  c0_vec_tk : numpy vector, shape (n_constraints,)
    Numpy vector encoding a part of the feasibility condition of this vertex in the
      'tk' chart.
    Only defined for finite (non-singular) vertices.
    Not always defined, computed and stored when used in feasibility tests.
  neighbors_dict : dictionary
    The dictionary for dom_regime neighbors of this dom_regime.
    Has four keys, three of them are "finite", "infinite", and "all",
      for finite, infinite (ray) dom_regime neighbors, and all of dom_regime
      neighbors, respectively.
    For each of these keys, we get a dictionary as well, with a (perm,row_idx)
      tuple as key and the dom_regime object as value.
    The last key is "zero", which maps to neighbors connected via dom_regimes
      that are zero rays in reaction orders (or log derivative).
      The value is itself a dictionary with "finite", "infinite",
      and "all" mapping to dictionaries with (ld:ld_regime) entries.
  neighbors_constrained_dict : dictionary
    The dictionaory for ld_regime neighbors of this ld_regime that are feasible
      under constraints applied to each dom_regime. 
    Same keys as neighbors_dict.
  """
  def __init__(self,perm,bn):
    """Initiates a ROP vertex

    Parameters
    ----------
    perm : an integer tuple
      an integer tuple of length dim_d, indicating for each conserved quantity, which species is dominant.

    bn : a binding network object
      the binding network that this vertex is a part of.
      used to get dimensions and stoichiometry matrix.
    """
    self.perm=perm
    self.bn=bn
    p_mat=np.zeros((bn.dim_d,bn.dim_n)) # p_mat is all 0 and 1
    p0_vec=np.zeros(bn.dim_d) # the value of nonzero entries of l_mat chosen by p_mat in each row for this vertex
    for i in range(bn.dim_d):
      p_mat[i,perm[i]]=1
      p0_vec[i]=np.log10(bn.l_mat[i,perm[i]])
    self.p_mat=p_mat
    self.p0_vec=p0_vec
    self.m_mat=np.concatenate((self.p_mat,self.bn.n_mat),axis=0)
    self.m0_vec=np.concatenate((self.p0_vec,np.zeros(self.bn.dim_r)),axis=0)
    self.orientation=np.sign(np.linalg.det(self.m_mat))
    self.is_feasible=True

  def vertex_ld_calc(self):
    # calculate log derivative matrix,
    # gives ray direction matrix if infinite vertex
    dim_n = self.bn.dim_n
    m_mat = self.m_mat
    if self.orientation==0: # if singular, get the ray direction
      m_mult_left=np.kron(np.eye(dim_n),m_mat)
      m_mult_right=np.kron(m_mat.T,np.eye(dim_n))
      temp=np.concatenate( (m_mult_left,m_mult_right),axis=0)
      # the following assertion test is not necessary if guaranteed to be rank 1 singularity.
      # assert temp.shape[1]-np.linalg.matrix_rank(temp)==1, "the vertex is singular in more than one direction."
      # we assert that an infinite vertex cannot be singular in more than one direction
      # so the ray is always rank 1. We assume higher order rays are convex combinations of first order rays.
      rslt=null_space(temp)
      # the following normalizes the entries to largest entry = 1.
      h_mat_coarse=np.reshape(rslt,(dim_n,dim_n),order='F') #order='F' is earlier index changes first. here fortran
      temp=np.abs(h_mat_coarse)
      minfactor=np.min(temp[temp>1e-7])
      h_mat_coarse_int=h_mat_coarse/minfactor #all non-zero entries are now integers
      h_mat_int=np.rint(h_mat_coarse_int)
      assert np.max(np.abs(h_mat_int-h_mat_coarse_int)) <1e-3, 'rounding of h_mat for vertices caused large error.'
      h_mat=h_mat_int/np.max(h_mat_int) # normalize largest entry to 1.

      # now we need to check the directionality. This utilizes neighbors
      vv_nb=list(self.neighbors_dict['finite'].values())[0]
      if np.sign(np.linalg.det(vv_nb.h_mat+1e5*h_mat))!=self.bn.orientation:
        h_mat=-1*h_mat
    else: # if not singular, just invert the matrix.
      h_mat = np.linalg.inv(m_mat)
      self.h0_vec = -h_mat.dot(self.m0_vec) # only non-singular vertices have h0_vec.
    self.h_mat=h_mat

  def vertex_c_mat_x_calc(self):
    # Get feasibility condition for this vertex expressed in (x) chart.
    # c_mat*log(x) + c0_vec > 0 is the validity condition, log is log10.
    l_mat=self.bn.l_mat
    j_list=self.j_func(self.p_mat,l_mat)
    dim=np.sum([len(j) for j in j_list])
    c_mat_x=np.zeros((dim,self.bn.dim_n))
    c0_vec = np.zeros(dim)
    dim_n=self.bn.dim_n
    counter=0
    for i in range(self.bn.dim_d):
      j_i=self.perm[i]
      for j in j_list[i]:
        c_mat_x[counter,j_i]=1
        c_mat_x[counter,j]=-1
        c0_vec[counter]=np.log10(l_mat[i,j_i])-np.log10(l_mat[i,j])
        counter+=1
    self.c_mat_x=c_mat_x
    self.c0_vec = c0_vec

  def j_func(self,p_mat,l_mat):
    """
    For p_mat and l_mat, find (column) indices that are nonzero in each row of
      l_mat but not nonzero in p_mat.

    Parameters
    ----------
    p_mat : numpy array
      d-by-n matrix, each row has only one nonzero entry that is 1.
    l_mat : numpy array
      d-by-n matrix, binding network's conservation law matrix.

    Returns
    -------
    j_list: list of int
      List of (column) indices that are nonzero in l_mat but zero in p_mat.
    """
    j_list=[]
    l_mat_masked=np.ma.array(l_mat,mask=p_mat).filled(fill_value=0)
    for i in range(p_mat.shape[0]):
      j_list+=[np.nonzero(l_mat_masked[i,:])[0]]
    return j_list

  def vertex_feasibility_test(self,chart='x',opt_constraints=[],positive_threshold=1e-5,is_asymptotic=True):
    """
    For given constraints, compute whether this vertex is feasible.

    Parameters
    ----------
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'.
    opt_constraints : list of cvxpy inequalities, optional
      A list of optimization constraints specified in terms of inequalities
        relating cvxpy variables of the binding network.
    positive_threshold : float, optional
      The vertex itself has inequality conditions, of the form c_mat*x + c0_vec > th,
        where th is the positive threshold used here. Default to 1e-5.
    is_asymptotic : boolean, optional
      Whether the inequalities of the vertex itself should be considered asymptotically
        or exactly. If asymptotically, then the inequality tested ommits c0_vec,
        so it is c_mat*x > th. is_asymptotic=True corresponds to inequality satisfied
        for the positive projective measure (where a ray is an infinitesimal of volume),
        and is_asymptotic=False is Lebesgue measure (a point is an infinitesimal of volume).

    Returns
    -------
    is_feasible: boolean
       whether this vertex is feasible under the constraints.
    """
    # first prepare the c_mat and c0_vec for the desired chart.
    c_mat,c0_vec=self.chart_check(chart)
    is_feasible=True
    opt_var=self.bn.opt_var #opt_var came from the binding network.
    if is_asymptotic: #if asymptotic, c0_vec is zero.
      prob=cp.Problem(cp.Minimize(1),opt_constraints+[c_mat @ opt_var >= positive_threshold])
    else: # if not asymptotic, it is exact, so c0_vec is taken into account.
      prob=cp.Problem(cp.Minimize(1),opt_constraints+[c_mat @ opt_var + c0_vec >= positive_threshold])
    prob.solve()
    if prob.status=='infeasible':
      is_feasible=False
    return is_feasible

  def vertex_find_neighbors(self):
    # Given a binding network with (already-computed) vertex dictionaries,
    #   compute this vertex's neighbors and store in self.neighbors,
    #   depending on whether it is finite or infinite.
    # Neighbor is defined as changing one row by moving the "1" to another place.
    # For chart 'x', neighbor is more nuanced.
    #   For example, (0,7,7) can have a neighbor (7,1,7) because it has 
    #     neighbor (7,7,7), which is shared with (7,1,7).
    #   But we do not consider (7,7,7) as an "infinite vertex",
    #     since >1 order infinite vertices only matter as a 
    #     "connecting region", and they are ignored in vertex construction. 
    #   To include these cases, for an infinite vertex perm1, for a given candidate 
    #     infinite neighbor perm 2, if difference is 2 then of course it's a neighbor,
    #     e.g. (5,5,2) and (5,5,3); if difference is more than 2, then switch the
    #     differing rows' order and see whether now they match with difference in just one row.
    #   Since whenever this is the case, there is the regime (j,j,j,*)
    neighbors_fin={}
    neighbors_inf={}
    for perm,vv in self.bn.vertex_dict['finite'].items():
      # temp=[not np.all(self.p_mat[i,:]==vv.p_mat[i,:]) for i in range(self.bn.dim_d)]
      # if np.sum(temp)==1: #the difference is just one row
      if np.sum(np.abs(self.p_mat-vv.p_mat))==2: #difference is in just one row and it's -1, 1.
        neighbors_fin[perm]=vv
    for perm,vv in self.bn.vertex_dict['infinite'].items():
      # temp=[not np.all(self.p_mat[i,:]==vv.p_mat[i,:]) for i in range(self.bn.dim_d)]
      # if np.sum(temp)==1: #the difference is just one row
      if np.sum(np.abs(self.p_mat-vv.p_mat))==2: #difference is in just one row and it's -1, 1.
        neighbors_inf[perm]=vv
    
    # if self.orientation==0:
    neighbors_all={**neighbors_fin,**neighbors_inf}
    self.neighbors_dict={'finite':neighbors_fin,'infinite':neighbors_inf,'all':neighbors_all}


  def vertex_update_constrained_neighbors(self):
    # Using vertices' is_feasible tag to have updated neighbors under constraints.
    # The feasible neighbors of an infeasible neighbor becomes this vertex's neighbors
    # under constraints.
    # If an infeasible neighbor has infeasible neighbors as well, we need to look at
    # their feasible neighbors as neighbors under constraint as well.
    # So this uses recursion.
    # The resulting constrained neighbors is stored in self.neighbors_constrained_dict.

    def extract_infeasible_neighbor(neighbors_all,neighbors_feasible_all,visited):
      infeasible_neighbors = {
        perm: vv for perm,vv in neighbors_all.items() if not vv.is_feasible
      }
      if infeasible_neighbors:  # the dictionary is not empty
        for perm,vv in infeasible_neighbors.items():
            if perm not in visited:
                visited.add(perm)
                neighbors_feasible_all.update({perm:vv for perm,vv in vv.neighbors_dict['all'].items() if vv.is_feasible})
                neighbors_feasible_all, visited = extract_infeasible_neighbor(
                  vv.neighbors_dict['all'],
                  neighbors_feasible_all,
                  visited
                )
      return neighbors_feasible_all, visited

    # first, each neighbor that is feasible should be a neighbor under constraint.
    neighbors_feasible_all = {perm:vv for perm,vv in self.neighbors_dict['all'].items() if vv.is_feasible}
    visited = set()

    neighbors_feasible_all,visited = extract_infeasible_neighbor(
      self.neighbors_dict['all'],neighbors_feasible_all,visited
    )
    # Remove self from the neighbors' dictionaries if present
    try:
        del neighbors_feasible_all[self.perm]
    except KeyError:
        pass
    neighbors_feasible_inf = {
        perm:vv for perm,vv in neighbors_feasible_all.items() if vv.orientation==0
        }
    neighbors_feasible_fin = {
        perm:vv for perm,vv in neighbors_feasible_all.items() if vv.orientation!=0
        }
    self.neighbors_constrained_dict={'finite':neighbors_feasible_fin,
                                  'infinite':neighbors_feasible_inf,
                                  'all':neighbors_feasible_all}

  def vertex_c_mat_xak_calc(self):
    # get the C matrix but for log(xa,k) coordinate
    # store as self.c_mat_xak
    try: c_mat_x=self.c_mat_x
    except AttributeError:
      self.vertex_c_mat_x_calc()
      c_mat_x=self.c_mat_x
    try: xak2x_map=self.bn.xak2x_map
    except AttributeError:
      self.bn.calc_xak2x_map()
      xak2x_map=self.bn.xak2x_map
    c_mat_xak=c_mat_x.dot(xak2x_map)
    self.c_mat_xak=c_mat_xak

  def vertex_c_mat_tk_calc(self):
    # get the C matrix but for log(t,k) coordinate
    # store as self.c_mat_tk
    assert self.orientation!=0, "only finite vertices can have non-singular (t,k) chart"
    try:
      c_mat_x=self.c_mat_x
    except AttributeError:
      self.vertex_c_mat_x_calc()
      c_mat_x=self.c_mat_x
    # The equivalence here is the following:
    # c_mat_tk @ log(t,k) + c0_vec_tk >=0  <=>  c_mat_x @ logx + c0_vec >=0
    c_mat_tk=c_mat_x.dot(self.h_mat) #because h_mat*log(t,k)=log(x)
    c0_vec_tk=self.c0_vec - c_mat_x.dot(self.h_mat.dot(self.m0_vec))
    self.c_mat_tk=c_mat_tk
    self.c0_vec_tk=c0_vec_tk

  def chart_check(self,chart):
    """
    Prepare the c_mat and c0_vec for the desired chart.

    Parameters
    ----------
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'.
    
    Returns
    ---------
    c_mat : ndarray, shape (n_constraints, n_var)
      matrix used in this vertex's feasibility condition in the 
        desired chart, e.g. it is c_mat * x + c0_vec > 0 in chart 'x'.
    c0_vec : ndarray vector, shape (n_constraints,)
      The vector used in this vertex's feasibility condition in the
        desired chart, e.g. it is c_mat * x + c0_vec > 0 in chart 'x'.
    """
    if chart=='x':
      try:
        c_mat=self.c_mat_x
        c0_vec=self.c0_vec
      except AttributeError: # if c_mat_x is not yet calculated
        self.vertex_c_mat_x_calc()
        c_mat=self.c_mat_x
        c0_vec=self.c0_vec
    elif chart=='xak':
      try:
        c_mat=self.c_mat_xak
        c0_vec=self.c0_vec #c0_vec is the same for chart x and xak.
      except AttributeError: # if c_mat_xak is not yet calculated
        self.vertex_c_mat_xak_calc()
        c_mat=self.c_mat_xak
        c0_vec=self.c0_vec
    elif chart=='tk':
      try:
        c_mat=self.c_mat_tk
        c0_vec=self.c0_vec_tk
      except AttributeError: # if c_mat_xak is not yet calculated
        self.vertex_c_mat_tk_calc()
        c_mat=self.c_mat_tk
        c0_vec=self.c0_vec_tk
    else:
      raise Exception('chart that is not one of x,xak or tk is not implemented yet')
    return c_mat,c0_vec

  def vertex_hull_of_validity(self,chart='x',margin=0,logmin=-6,logmax=6,c_mat_extra=[],c0_vec_extra=[]):
    """
    Compute the vertices of the validity region as a bounded convex hull.

    Parameters
    ----------
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'.
    margin : float, optional
      The vertex's feasibility conditions are inequalities, 
        of the form c_mat*x + c0_vec >= margin (e.g. in 'x' chart),
        Margin defaults to 0, and its values are in log10.
      This can be adjusted to be stronger/weaker requirements on dominance.
    logmin : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    logmax : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    c_mat_extra : ndarray, shape (n_constraints, n_var)
      Extra optimization constraints to be added to feasibility conditions,
        in the form of c_mat_extra @ var + c0_vec_extra >= 0.
    c0_vec_extra : numpy vector, shape (n_constraints,)
      Extra optimization constraints to be added to feasibility conditions,
        in the form of c_mat_extra @ var + c0_vec_extra >=0.
      
    Returns
    -------
    points : ndarray
      The points corresponding to vertices of the convex hull that is the 
        region of validity.
    feasible_point: ndarray vector
      The point that is feasible in the interior of the convex hull.
    hs : scipy.spatial.HalfspaceIntersection object
      The half space intersection built from the feasibility inequalities.
    """
    
    # first check whether logmin and logmax are scalars or vectors.
    try: 
      float(logmin) # if logmin and logmax are scalars
      bbox = np.repeat(np.array([[logmin,logmax]]),self.bn.dim_n,axis=0) #bounding box to make polyhedra bounded
    except TypeError: # if logmin and logmax are vectors
      # stack them horizontally as column vectors
      bbox = np.hstack((logmin[:,None],logmax[:,None]))

    c_mat,c0_vec=self.chart_check(chart)

    # the inequality c_mat*var + c0_vec - margin >= 0, becomes A*var + b <=0
    # where A = -c_mat, and b = th - c0_vec.
    # With extra constraints, margin is always 0 for extra constraints,
    # so margin_full = vstack((margin,zeros)) where zeros is of length len(c0_vec_extra)
    if np.any(c_mat_extra): # if there are additional constraints
      c_mat_full=np.vstack((c_mat,c_mat_extra))
      c0_vec_full=np.concatenate((c0_vec,c0_vec_extra))
      margin_vec_full = np.concatenate((margin*np.ones(c0_vec.shape[0]),np.zeros(c0_vec_extra.shape[0])))
    else: # there are no additional constraints
      c_mat_full=c_mat 
      c0_vec_full = c0_vec
      margin_vec_full = margin*np.ones(c0_vec.shape[0])
    A=-c_mat_full # negtive because the optimization code is for Ax+b<=0, while our notation is c_mat*x+c0_vec >=0.
    b=margin_vec_full - c0_vec_full

    points, hull, feasible_point, hs = self.get_convex_hull(A,b,bbox)
    return points, hull, feasible_point,hs

  def __feasible_point_calc(self,A, b):
    # Finds the center of the largest sphere fitting in the convex hull of
    #   A x + b <= 0.
    # Use method in description of scipy.spatial.HalfspaceIntersection.
    # Based on Chebyshev center finding of Boyd's book 4.3.1
    # Needs to solve max y s.t. A x + y |A_i| + b <= 0. A_i are rows of A,
    #   so |A_i| is the norm vector of rows of A.
    # We transform this inequality into min c*x, A_lp x <= b_lp,
    #   standard form of a linear program,
    #   where c=(0,...,0,-1) so that c*x = x[-1] = y, and x[:-1] is x above,
    #   A_lp = hstack(A,|A_i|). b_lp is the same as -b, but as a column vector.
    norm_vector = np.linalg.norm(A, axis=1) # Frobenius norm
    A_linprog = np.hstack((A, norm_vector[:, None])) 
    b_linprog = -b[:, None] # this makes b into shape len(b)-by-1.
    c=np.zeros((A.shape[1] + 1,))
    c[-1] = -1
    res = linprog(c, A_ub=A_linprog, b_ub=b_linprog, bounds=(None, None))
    if res.status!=0: breakpoint()
    return res.x[:-1]

  def __add_bbox(self,A, b, bbox):
    # in case A x + b <= 0 is not bounded, add a bounding box specified by bbox.
    # bbox is an array, n-by-2, the ith row is (min,max) of x_i.
    # Transform: bbox[i,0] is min, so x_i >= bbox[i,0] becomes -x_i + bbox[i,0] <=0.
    #   This is encoded in A's entry is -I, and b's entry is bbox[i,0].
    #   Similarly for max, A's entry is +I, and b's entry is -bbox[i,1].
    dim_n=A.shape[1]
    A_bounded=A
    b_bounded=b
    for i in range(dim_n):
      A_bounded = np.vstack((A_bounded,-np.eye(1,dim_n,i),np.eye(1,dim_n,i)))
      b_bounded = np.hstack((b_bounded,bbox[i,0],-bbox[i,1]))
    return A_bounded, b_bounded   
  
  def __hs_intersection(self,A, b, feasible_point):
    # HalfspaceIntersection take the convention halfspaces=[A;b]
    #   to indicate A x + b <= 0.
    halfspaces = np.hstack((A, b[:, None]))
    # hs = HalfspaceIntersection(halfspaces, feasible_point,qhull_options='QJ') #QJ option to joggle to avoid non-full-dimensional constraints.
    # Qt option to triangulate all the time to avoid precision issues.
    # This is helpful to avoid problems since later on the "ConvexHull"
    #   function uses this Qt option.
    hs = HalfspaceIntersection(halfspaces, feasible_point,qhull_options='Qt') 
    return hs


  def get_convex_hull(self,A_local, b_local, bbox):
    # Given A,b for halfspace intersection A x + b <=0,
    #   and bounding box bbox,
    #   get the vertices of the convex hull that formed.
    # Modified from https://stackoverflow.com/questions/65343771/solve-linear-inequalities
    A_bounded, b_bounded = self.__add_bbox(A_local, b_local, bbox)
    feasible_point = self.__feasible_point_calc(A_bounded, b_bounded)
    hs = self.__hs_intersection(A_bounded, b_bounded, feasible_point)
    # hs = hs_intersection(A, b, interior_point)
    points = hs.intersections
    hull = ConvexHull(points,qhull_options='Q12') # to allow wide facets and dulbridge... meaning what???
    return points,points[hull.vertices],feasible_point, hs


  def vertex_hull_sampling(self,nsample,chart='x', margin=0,logmin=-6,logmax=6,c_mat_extra=[],c0_vec_extra=[]):
    """
    Sample points in the vertex's region of validity based on its hull of feasible regions.
    Extra conditions (linear) can be added as c_mat_extra @ var + c0_vec_extra >= 0.

    Parameters
    ----------
    nsample : int
      Number of points to be sampled.
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'.
    margin : float, optional
      The vertex's feasibility conditions are inequalities, 
        of the form c_mat*x + c0_vec >= margin (e.g. in 'x' chart),
        where margin is the margin used here. Default to 0.
      This can be adjusted to be stronger/weaker requirements on dominance.
    logmin : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    logmax : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    c_mat_extra : ndarray, shape (n_constraints, n_var)
      Extra optimization constraints to be added to feasibility conditions,
        in the form of c_mat_extra @ var + c0_vec_extra >= 0.
    c0_vec_extra : numpy vector, shape (n_constraints,)
      Extra optimization constraints to be added to feasibility conditions,
        in the form of c_mat_extra @ var + c0_vec_extra >=0.

    Returns
    -------
    sample : ndarray of shape nsample-by-dim_n
      dim_n is number of species in the binding network.
      Sampled points satisfying the feasibility conditions of this vertex.
      Each row (sample[i,:]) is a sampled point.
    """
    # first compute the convex hull for this vertex's validity and get the vertex points.
    points,hull,_,_=self.vertex_hull_of_validity(chart=chart,margin=margin,logmin=logmin,logmax=logmax,c_mat_extra=c_mat_extra,c0_vec_extra=c0_vec_extra)
    sample=self.__dist_in_hull(points,nsample,points_are_vertices=False)
    # To sample a simplex in n-dim uniformly, take n uniform(0,1) random 
    #   variables and take difference after padding 0 at the beginning and
    #   1 at the end. This gives a vector of n-dim in the simplex, with
    #   a probability density that is uniform in the simplex.
    #   See https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # ncoeffs=points.shape[0]
    # temp=np.sort(np.random.rand(nsample,ncoeffs-1),axis=1)
    # coeffs=np.diff(temp,prepend=0,append=1,axis=1)
    # sample=coeffs@points # this has shape nsample-by-dim_n
    return sample

  def __dist_in_hull(self,points, nsample, points_are_vertices=False):
    """
    Create uniform sample over convex hulls by Delaunay triangulation.
    Adapted from https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    
    Parameters
    ----------
    points : ndarray, shape (num_points,dim).
      Points whose convex hull is to be sampled uniformly.
    nsample : int
      Number of points to be sampled.
    points_are_vertices : bool, optional
      If True, points are assumed to be vertices.
      If False, convex hull of points is first taken to find vertices.
      Defaults to False.

    Returns
    -------
    sample : ndarray, shape (nsample, dim).
      sampled points uniformly in convex hull of points.
    """
    dims = points.shape[-1]
    if points_are_vertices:
      hull=points 
    else: 
      hull = points[ConvexHull(points,qhull_options='Q12').vertices]
    breakpoint()
    deln = hull[Delaunay(hull,qhull_options='Q12').simplices]
    vols = np.abs(np.linalg.det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = nsample, p = vols / vols.sum())
    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = nsample))

  def vertex_print_validity_condition(self,is_asymptotic=False):
    """
    print the expression for t=x, x(t,k) and inequalities for the
    region of validity given a regime, using the labels of x,t,k.

    Parameters
    ----------
    is_asymptotic : boolean, optional
      If False, the inequalities are shown with intercept.
      If True, the inequalities are shown without intercept.

    Returns
    -------
    None, but print a lot of text.
    """
    try:
      c_mat=self.c_mat_tk
      c0_vec=self.c0_vec_tk
    except AttributeError: # if c_mat_xak is not yet calculated
      self.vertex_c_mat_tk_calc()
      c_mat=self.c_mat_tk
      c0_vec=self.c0_vec_tk
    h_mat=self.h_mat
    h0_vec=self.h0_vec
    p0_vec=self.p0_vec
    perm=self.perm
    x_sym=self.bn.x_sym
    tk_sym=self.bn.tk_sym

    print('======This is vertex perm '+str(perm)+'======\n')
    t2x_list=[]
    for i in range(self.bn.dim_d):
      p0=p0_vec[i]
      if p0==1 or is_asymptotic:
        t2x_list+=[tk_sym[i].name + '=' + x_sym[perm[i]].name]
      else:
        t2x_list+=[tk_sym[i].name + '=' + str(p0) + x_sym[perm[i]].name]
    print('(1) t in x: t is dominated by... \n'+ ','.join(t2x_list) + '\n')

    # BELOW we obtain x's expression in terms of t,k by
    # log(x) = h_mat * log(t,k) + h0_vec

    if self.orientation==0:
      print('(2) x in tk: Vertex is singular, cannot express (x) in (t,k).')
    else:
      x2tk_list=[]
      for i in range(self.bn.dim_n):
        h_row=h_mat[i,:]
        pos_idx=np.where(h_row>0)[0]
        neg_idx=np.where(h_row<0)[0]
        expr_pos=' '.join([tk_sym[j].name + '^' + str(h_row[j]) if h_row[j]!=1.0 else tk_sym[j].name for j in pos_idx])
        expr_neg=' '.join([tk_sym[j].name + '^' + str(-h_row[j]) if -h_row[j]!=1.0 else tk_sym[j].name for j in neg_idx])
        if neg_idx.size==0:
          expr=expr_pos
        else:
          expr=expr_pos + ' / ' + expr_neg
        h0=h0_vec[i]
        if h0==0:
          x2tk_list+=[x_sym[i].name + ' = '+ expr]
        else:
          x2tk_list+=[x_sym[i].name + ' = ' + expr + ' / '+str(10**h0)]
      print('(2) x in tk: \n'+',\n'.join(x2tk_list) + '\n')


    ineq_list=[]
    for i in range(c_mat.shape[0]):
      c_row=c_mat[i,:]
      pos_idx=np.where(c_row>0)[0]
      neg_idx=np.where(c_row<0)[0]
      expr_pos=' '.join([tk_sym[j].name + '^'+str(c_row[j]) if c_row[j]!=1.0 else tk_sym[j].name for j in pos_idx])
      expr_neg=' '.join([tk_sym[j].name + '^'+str(-c_row[j]) if -c_row[j]!=1.0 else tk_sym[j].name for j in neg_idx])
      c0=c0_vec[i]
      if c0==0:
        ineq_list+=[expr_pos + ' > ' + expr_neg]
      else:
        ineq_list+=[expr_pos + ' > ' + str(10**(-c0)) + ' ' + expr_neg]

    print('(3) BELOW is constraints for regions of validity \n'+',\n'.join(ineq_list) + '\n')


class binding_network:
  """A binding network object.

  Parameters
  ----------
  n_mat : numpy array
    The stoichiometry matrix defining the binding network.
  is_atomic: 'bool'
    Is the binding network atomic or not.
  l_mat : numpy array
    The conservation law matrix defining the conserved total quantities.
    If the network is atomic, it can be directly computed from n_mat.
  l_mat_sym

  dim_n : 'int'
    The number of chemical species, same as number of columns of n_mat and l_mat.
  dim_r : 'int'
    The number of (linearly independent) binding reactions, same as number of rows of n_mat.
  dim_d : 'int'
    The number of conserved quantities or totals, same as the number of rows of l_mat.
  x_sym : list of symbols
    An ordered list of symbols for the chemical species, denoting columns of the n_mat and l_mat.
  t_sym : list of symbols
    An ordered list of symbols for the total or conserved quantities, denoting rows of the l_mat.
  k_sym : list of symbols
    An ordered list of symbols for the binding constants in the dissociation direction, denoting rows of the n_mat.
  tk_sym : list of symbols
    An ordered list of symbols concatenating (t_sym, k_sym).
    Just for convenience when (t_sym, k_sym) needs to be called together.
  opt_var : list of cvxpy variables of length dim_n
    The cvxpy variables to be used for testing for feasibility etc in 
      optimization problems.
  m_mat : numpy array, n-by-n
    Concatenation of l_mat and n_mat vertically.
  orientation : 'int'
    Take value from {-1,0,+1}.
    This is the sign of m_mat's determinant.
  activity_regime_dict : dictionary
    Dictionary of dom_regimes for various activities on top of the 
      binding network.
    Keys are b_vec, the vectors defining different catalytic activities
      on top of the binding network.
    The value are dictionaries of dom_regimes, with keys 'finite', 
      'infinite', and 'all', and values are {(perm,row_idx):dom_regime}
      pairs.
    For example, activity_regime_dict[b_vec]['all'][(perm,row_idx)]
      yields a dom_regime object.
    Initializes as an empty dictionary.
    Computed by calling self.activity_regime_construct(b_vec).
  activity_ld_regime_dict : dictionary
    Dictionary of ld_regimes for various activities on top of the
      binding network.
    Keys are b_vec, the vectors defining different catalytic activities
      on top of the binding network.
    The value are dictionaries of dom_regimes, with keys 'finite', 
      'infinite', and 'all', and values are {ld:ld_regime} pairs.
    For example, activity_ld_regime_dict[b_vec]['all'][(perm,row_idx)]
      yields a dom_regime object.
    Initializes as an empty dictionary.
    Computed by calling self.activity_regime_construct(b_vec).
  activity_regime_constrained_dict : dictionary
    Dictionary of dom_regimes feasible under given constraints for 
      various activities.
    Same format as activity_regime_dict.
    Initializes as an empty dictionary.
    Computed by calling self.activity_regime_construct(b_vec,opt_constraints).
    Once a new opt_constraints is used, this dict is overwritten.
  activity_ld_regime_constrained_dict
    Dictionary of ld_regimes feasible under given constraints for 
      various activities.
    Same format as activity_regime_dict.
    Initializes as an empty dictionary.
    Computed by calling self.activity_regime_construct(b_vec,opt_constraints).
    Once a new opt_constraints is used, this dict is overwritten.
  """

  def __init__(self,
               n_mat,
               l_mat=np.array([]),
               is_atomic=False,
               x_sym=np.array([]),
               t_sym=np.array([]),
               k_sym=np.array([])):
    """Initiate a binding network.

    Parameters
    ----------
    n_mat : numpy array
        The stoichiometry matrix defining the binding network.
    is_atomic: 'bool', optional
        Is the binding network atomic or not.
    l_mat : numpy array, optional
        The conservation law matrix defining the conserved total quantities.
        If not given, and is_atomic is True, then it will be computed 
          from n_mat.
    x_sym : list of symbols, optional
        An ordered list of symbols for the chemical species, denoting columns of the n_mat and l_mat.
    t_sym : list of symbols, optional
        An ordered list of symbols for the total or conserved quantities, denoting rows of the l_mat.
    k_sym : list of symbols, optional
        An ordered list of symbols for the binding constants in the dissociation direction, denoting rows of the n_mat.
    """
    self.is_atomic=is_atomic
    self.n_mat=n_mat
    self.n_mat_sym=sp.Matrix(n_mat)
    self.dim_r,self.dim_n=self.n_mat.shape
    self.dim_d = self.dim_n - self.dim_r
    if not np.any(l_mat): #no input l_mat
      l_mat=self.l_from_n(self.n_mat)
      assert np.all(l_mat>=0)
    # if no input symbols for the x,t,k variables then give them default numerically ordered ones.
    if not np.any(x_sym):
      x_sym = sp.symbols("x:"+str(self.dim_n))
    if not np.any(t_sym):
      t_sym = sp.symbols("t:"+str(self.dim_d))
    if not np.any(k_sym):
      k_sym = sp.symbols("k:"+str(self.dim_r))

    self.l_mat=l_mat
    self.l_mat_sym=sp.Matrix(l_mat)
    self.x_sym=x_sym
    self.t_sym=t_sym
    self.k_sym=k_sym
    self.tk_sym=t_sym+k_sym
    self.opt_var=cp.Variable(self.dim_n)

    self.m_mat=np.concatenate((self.l_mat,self.n_mat),axis=0)
    self.orientation=np.sign(np.linalg.det(self.m_mat))

    self.ld_sym_dict={}
    self.activity_regime_dict={}
    self.activity_ld_regime_dict={}
    self.activity_regime_constrained_dict={}
    self.activity_ld_regime_constrained_dict={}

  def l_from_n(self,n_mat):
    """if the network is atomic, compute the L matrix from the N matrix

    Parameters
    ----------
    n_mat : numpy array
        The stoichiometry matrix defining the binding network.
        Assumes the columns are ordered so that the atomic species come first.

    Returns
    -------
    l_mat: numpy array
       The conservation law or totals matrix.
    """
    assert self.is_atomic
    r=n_mat.shape[0]
    d=n_mat.shape[1]-r
    n1_mat=n_mat[:,:d]
    n2_mat=n_mat[:,-r:]
    l2_mat=-(n1_mat.T).dot(np.linalg.inv(n2_mat).T)
    l_mat=np.concatenate((np.eye(d),l2_mat),axis=1)
    return l_mat

  def logder_num(self,logvar,chart='x',a_mat=np.array([])):
    """compute the numerical log derivative of the binding network at points 
      specified by logvar in specified chart and dominance a_mat.

    Parameters
    ----------
    logvar : ndarray n_points-by-dim_n
      Array of the points to evaluate the log derivatives at, in base-10 log.
      In chart 'x', for example, this is logx. 
    chart : str
      Specifying the chart that logvar is specified in, could be 'x','xak','tk'.
    a_mat : numpy array, optional
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.
      Optional, defaults to l_mat of the binding network.

    Returns
    -------
    logder: ndarray, shape (n_points,dim_n,dim_n)
      array of n-by-n matrix of log derivative of x to (t,k), where 
        t=a_mat@x, n is number of species.
    logx : ndarray, shape (n_points,dim_n)
      array of logx that the logvar points correspond to.
        This is returned since all input var, regardless of chart,
        is mapped to logx chart first. 
        So we also return this for convenience.
    """
    # first check a_mat makes sense.
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    else:
      assert a_mat.shape==(self.dim_d,self.dim_n), f"the shape of L matrix should be {self.dim_d} by {self.dim_n}."
      assert np.all(a_mat>=0), "all entries of A matrix should be non-negative."
      assert np.all(a_mat.dot(np.ones(self.dim_n))>0), "each row of A matrix should have at least one positive entry."
    # for different charts, use different functions to evaluate
    npts=logvar.shape[0]
    assert logvar.shape[1]==self.dim_n, 'shape of logvar should be num_points-by-dim_n'
    logders=np.empty((logvar.shape[0],self.dim_n,self.dim_n))
    if chart=='x':
      for i in range(npts):
        logders[i]=self.logder_x_num(logvar[i],a_mat)
      logx=logvar
    elif chart=='xak':
      assert self.is_atomic, 'the binding network is not atomic, cannot use xak chart'
      logx=np.empty(logvar.shape)
      for i in range(npts):
        logders[i],logx[i]=self.logder_xak_num(logvar[i],a_mat)
    elif chart=='tk':
      logx=np.empty(logvar.shape)
      for i in range(npts):
        logders[i],logx[i]=self.logder_tk_num(logvar[i],a_mat)
    else: 
      raise Exception('chart that is not one of "x,xak,tk" is not implemented yet')
    return logders,logx

  def logder_x_num(self,logx,a_mat):
    """compute the numerical log derivative of the binding network at one point in chart x.

    Parameters
    ----------
    logx : numpy vector
      Vector of concentrations for all the species in log, base-10.
    a_mat : numpy array
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.

    Returns
    -------
    logder: ndarray, shape (n_points,dim_n,dim_n)
      array of n-by-n matrix of log derivative of x to (t,k), where 
        t=a_mat@x, n is number of species.
    """
    x=10**logx
    t_inv = 1/(a_mat.dot(x))
    temp=a_mat*x
    upper=(temp.T*t_inv).T
    logder_inv=np.concatenate((upper,self.n_mat),axis=0)
    try:
      np.linalg.inv(logder_inv)
    except:
      print(x)
      print(t_inv)
      print(logder_inv)
      
    return np.linalg.inv(logder_inv)

  def logder_xak_num(self,logxak,a_mat):
    """compute the numerical log derivative of dlog(x)/dlog(a_mat*x,k) at a point
    specified by log(xa,k), where xa is concentration of atomic species,
    k is binding constants. log is base 10.
    Assumes the network is atomic, and n_mat,a_mat have atomic species coming first.

    Parameters
    ----------
    logxak : numpy vector, shape (dim_n,)
      Vector numerical value for atomic species concentration (first dim_d 
        entries) and binding reaction constants (last dim_r entries). 
      log is base 10.
    a_mat : a_mat : numpy array.
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.

    Returns
    -------
    logder: ndarray, shape (n_points,dim_n,dim_n)
      array of n-by-n matrix of log derivative of x to (t,k), where 
        t=a_mat@x, n is number of species.
    logx : ndarray, shape (n_points,dim_n)
      array of logx that the logvar points correspond to.
        This is returned since all input var, regardless of chart,
        is mapped to logx chart first. 
        So we also return this for convenience.
    """
    ## commented out are old code that directly calculate logx, now we use stored map.
    # d=self.dim_d
    # logxa=logxak[:d]
    # logk=logxak[d:]
    # temp1=(a_mat.T).dot(logxa)
    # n2_mat=self.n_mat[:,d:]
    # temp2=np.linalg.inv(n2_mat).dot(logk)
    # logx=temp1 + np.pad(temp2,(d,0),mode='constant',constant_values=(0,0))

    logx=self.xak2x_num(logxak)
    return self.logder_x_num(logx,a_mat),logx

  def logder_tk_num(self,logtk,a_mat):
    """compute the numerical log derivative of dlog(x)/dlog(a_mat@x,k) at a point
    specified by log(t,k), where t=a_mat@x is concentration of atomic species,
    k is binding constants. log is base 10.

    Parameters
    ----------
    logtk : numpy vector, shape (dim_n,)
      Vector numerical value for total concentration (first dim_d 
        entries) and binding reaction constants (last dim_r entries). 
      log is base 10.
    a_mat : a_mat : numpy array.
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.

    Returns
    -------
    logder: ndarray, shape (n_points,dim_n,dim_n)
      array of n-by-n matrix of log derivative of x to (t,k), where 
        t=a_mat@x, n is number of species.
    logx : ndarray, shape (n_points,dim_n)
      array of logx that the logvar points correspond to.
        This is returned since all input var, regardless of chart,
        is mapped to logx chart first. 
        So we also return this for convenience.
    """
    logx=self.tk2x_num(logtk,a_mat)
    return self.logder_x_num(logx,a_mat),logx

  def logder_activity_num(self,b_vec,logx_array,ld_mat_array):
    """given a logder matrix, compute the logder of b^T x.

    Parameters
    ----------
    b_vec : numpy vector, shape (dim_n,)
      Vector indicating which species are included in the catalytic activity.
      All entries are non-negative, with at least one nonzero.
    logx_array : ndarray, shape (n_points,dim_n)
      array of logx-vector indicating the point at which the 
      logder is evaluated. log is base 10.
    ld_mat_array : ndarray, shape (n_points,dim_n,dim_n)
      array of n x n matrix for dlogx/dlog(t,k) at point x
        on the manifold.

    Returns
    -------
    ld_activity: ndarray, shape (n_points,dim_n)
       array of vectors for dlog(b^T x)/dlog(t,k)
    """
    assert np.all(b_vec>=0), "all entries of b_vec should be non-negative."
    assert np.sum(b_vec)>0, "there should be at least one nonzero entry in b_vec."
    x_array=10**logx_array # shape (n_points,dim_n)
    bx_array=x_array*b_vec # each row element-wise product with b_vec
    # so bx_array has shape (n_points,dim_n)
    coeff=(bx_array.T/np.sum(bx_array,axis=1)).T * np.log(10) # shape (n_points,dim_n)
    npts=logx_array.shape[0]
    
    # One implementation is iterate and sum, this is slow (not really?).
    ld_activity=np.empty((npts,self.dim_n))
    for i in range(npts):
      ld_activity[i]=coeff[i].dot(ld_mat_array[i])

    # Another implementation is directly writing out the product
    # coeff_temp=np.repeat(coeff[:,:,np.newaxis],self.dim_n,axis=2)
    # ld_activity=(ld_mat_array*coeff_temp).sum(-1)

    # Yet another way is to use einsum. Will do these if speed is a problem.

    return ld_activity

  def x2tk_num(self,logx):
    """compute the (logt,logk) value given logx

    Parameters
    ----------
    logx : numpy vector
      Vector numerical value for total variables that define the x point.

    Returns
    -------
    logt,logk: numpy vector
      The numerical value of x at this point. log is base 10.
    """
    logt=np.log10(self.l_mat.dot(10**logx))
    logk=self.n_mat.dot(logx)
    return logt,logk

  def xak2x_num(self,logxak):
    """compute the logx value given logxak=(logxa,logk)

    Parameters
    ----------
    logxak : numpy vector, shape (dim_n,)
      Vector numerical value for atomic species concentration (first dim_d 
        entries) and binding reaction constants (last dim_r entries). 
      log is base 10.

    Returns
    -------
    logx: numpy vector
      The numerical value of logx at this point.
    """
    try: xak2x_map=self.xak2x_map
    except AttributeError:
      self.calc_xak2x_map() # if doesn't exist, calculate it
      xak2x_map=self.xak2x_map
    return xak2x_map.dot(logxak)

  def calc_xak2x_map(self):
    # Defines the matrix (linear map) that takes log(xa,k) to log(x)
    # and stores it in self.xak2x_map
    d=self.dim_d
    r=self.dim_r
    l2_mat=self.l_mat[:,d:]
    n2_mat=self.n_mat[:,d:]
    upper=np.concatenate((np.eye(d),np.zeros((d,r))),axis=1)
    lower=np.concatenate((l2_mat.T,np.linalg.inv(n2_mat)),axis=1)
    temp=np.concatenate((upper,lower),axis=0)
    self.xak2x_map=temp

  def tk2x_num(self,logtk,a_mat):
    """compute the logx value by numerical integration along the equilibrium manifold 
    using log derivatives. The point on the manifold defined by logtk=(logt,logk) is
    the same as that defined by logx.

    Parameters
    ----------
    logtk : numpy vector, shape (dim_n,)
      Vector numerical value for total variables (in first dim_d entries) and 
       the binding reactino constants (in last dim_r entries) that define the point. 
       p_(logx) = p_(logt,logk). log is base 10.
    a_mat: numpy array
      The matrix defining the total variables t=a_mat@x that the log derivatives are taken with respect to.
      Default should be specified to be self.l_mat.

    Returns
    -------
    logx: numpy vector
      The numerical value of x at this point. log is base 10.
    """

    # the initial point is always x=1, (t,k) = (A*1, 1)
    # or, in log, logx=0, (logt,logk) = (log(A*1),0)

    logt0=np.log10(np.sum(a_mat,axis=1))
    logk0=np.zeros(self.dim_r)
    y0=np.concatenate((logt0,logk0),axis=0)
    y1=logtk
    logx0=np.zeros(self.dim_n)
    # The time is pseudo time, parameterizing trajectory from y0 to y1,
    # where y0=(logt0,logk0) = (log(A*1),0), and y1=(logt,logk) the input.
    # So a point on the trajectory is gamma(tau) = tau*(y1-y0)+y0, 0<=tau<=1.
    # The time derivative is therefore
    # dlogx/dtau (x0) = dlogx/dlog(t,k) (x0) * dlog(t,k)/dtau (x0)
    #                 = dlogx/dlog(t,k) (x0) * (y1-y0)
    # dlogx/dlog(t,k) (x0) is log derivative matrix evaluated at x0.
    time_derivative_func=lambda tau,logx: self.logder_x_num(logx,a_mat).dot(y1-y0)
    sol=solve_ivp(time_derivative_func, [0, 1], logx0)
    logx=sol.y[:,-1]
    return logx

  def logder_tk2x_num(self,logvar,chart='x',a_mat=np.array([])):
    """compute the numerical dlog(t,k)/dlog(x) log derivative of the binding network at points 
      specified by logvar in specified chart and dominance a_mat.

    Parameters
    ----------
    logvar : ndarray n_points-by-dim_n
      Array of the points to evaluate the log derivatives at, in base-10 log.
      In chart 'x', for example, this is logx. 
    chart : str
      Specifying the chart that logvar is specified in, could be 'x','xak','tk'.
    a_mat : numpy array, optional
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.
      Optional, defaults to l_mat of the binding network.

    Returns
    -------
    logder: ndarray, shape (n_points,dim_n,dim_n)
      array of n-by-n matrix of log derivative of (t,k) to x, where 
        t=a_mat@x, n is number of species.
    logx : ndarray, shape (n_points,dim_n)
      array of logx that the logvar points correspond to.
        This is returned since all input var, regardless of chart,
        is mapped to logx chart first. 
        So we also return this for convenience.
    """
    # first check a_mat makes sense.
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    else:
      assert a_mat.shape==(self.dim_d,self.dim_n), f"the shape of L matrix should be {self.dim_d} by {self.dim_n}."
      assert np.all(a_mat>=0), "all entries of A matrix should be non-negative."
      assert np.all(a_mat.dot(np.ones(self.dim_n))>0), "each row of A matrix should have at least one positive entry."
    # for different charts, use different functions to evaluate
    npts=logvar.shape[0]
    assert logvar.shape[1]==self.dim_n, 'shape of logvar should be num_points-by-dim_n'
    logders=np.empty((logvar.shape[0],self.dim_n,self.dim_n))
    if chart=='x':
      for i in range(npts):
        logders[i]=self.logder_tk2x_x_num(logvar[i],a_mat)
      logx=logvar
    elif chart=='xak':
      assert self.is_atomic, 'the binding network is not atomic, cannot use xak chart'
      logx=np.empty(logvar.shape)
      for i in range(npts):
        logx[i]=self.xak2x_num(logvar[i])
        logders[i]=self.logder_tk2x_x_num(logx[i],a_mat)
    elif chart=='tk':
      logx=np.empty(logvar.shape)
      for i in range(npts):
        logx[i]=self.tk2x_num(logvar[i],a_mat)
        logders[i]=self.logder_tk2x_x_num(logvar[i],a_mat)
    else: 
      raise Exception('chart that is not one of "x,xak,tk" is not implemented yet')
    return logders,logx

  def logder_tk2x_x_num(self,logx,a_mat):
    """compute the numerical dlog(t,k)/dlog(x) 
    log derivative of the binding network at one point in chart x.

    Parameters
    ----------
    logx : numpy vector
      Vector of concentrations for all the species in log, base-10.
    a_mat : numpy array
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.

    Returns
    -------
    logder: ndarray, shape (n_points,dim_n,dim_n)
      array of n-by-n matrix of log derivative of (t,k) to x, where 
        t=a_mat@x, n is number of species.
    """
    x=10**logx
    t_inv = 1/(a_mat.dot(x))
    temp=a_mat*x
    upper=(temp.T*t_inv).T
    logder=np.concatenate((upper,self.n_mat),axis=0)
    return logder
  
  def get_complex_ss(self, logk, logx):
    n1 = self.n_mat[:, : len(logx)]
    n2 = self.n_mat[:, len(logx): ]
    n2_inv = np.linalg.inv(n2)
    logx_complex = np.dot(n2_inv, (logk - np.dot(n1, logx)))

    return 10 ** logx_complex



# BELOW ARE VERTEX RELATED METHODS

  def vertex_construct(self):
    """
    Construct the rop_vertex objects that this binding network can have,
      compute their orientation and feasibility (without additional 
      constraints), and store them in self.vertex_dict.
    Then the vertices' neighbors, log derivative, and c_mat_xak
      are computed and stored in these objects.

    Parameters
    ----------
    None.

    Returns
    -------
    None. The vertices are recorded in self.vertex_dict and by updating
      the vertex objects.
    """
    # Construct a dictionary of reachable vertices.
    # because l_mat tends to be sparse, we iterate through its rows to get nonzero indices,
    # then each vertex's dominance condition a_mat is choosen from the nonezro indices.

    print('Constructing vertex objects and test for feasibility...')
    d,n=self.l_mat.shape
    orientation=self.orientation
    nnz_list=[]
    for i in range(d):
      nnz_list=nnz_list+[np.nonzero(self.l_mat[i,:])[0]]
    vertex_inf_dict={}
    vertex_fin_dict={}
    for perm in itertools.product(*nnz_list):
      vertex=rop_vertex(perm,self)
      if vertex.orientation==0:
        # this is possibly an infinite vertex
        # check for rank =1, meaning perm has exactly one repeat
        perm_count=[perm.count(perm[i]) for i in range(d)]
        if max(perm_count)==2:
          if perm_count.count(2)==2: # because that same value shows up twice, each with count 2.
            # after all that check, there still can be infinite vertices that are not reachable
            # so we do feasibility test.
            is_feasible=vertex.vertex_feasibility_test(chart='x',opt_constraints=[])
            # and then add to vertex_inf_dict.
            if is_feasible:
              vertex_inf_dict[perm]=vertex
      elif vertex.orientation==orientation:
        # this is a finite vertex with the right orientation
        # we test for feasibility
        is_feasible=vertex.vertex_feasibility_test(chart='x',opt_constraints=[])
        # and then add to vertex_fin_dict.
        if is_feasible:
          vertex_fin_dict[perm]=vertex
    vertex_dict={**vertex_fin_dict,**vertex_inf_dict}
    self.vertex_dict={'all':vertex_dict,'finite':vertex_fin_dict,'infinite':vertex_inf_dict}

    print('Finished vertex construction, now computing neighbors of vertices...')
    for perm,vv in vertex_dict.items():
      vv.vertex_find_neighbors()

    print('Finished neighbors, now computing log derivatives...')
    # first compute log der for finite vertices, since infinite ones rely on
    # finite neighbors to find orientation.
    for perm,vv in vertex_dict.items():
      vv.vertex_ld_calc()

    print('Finished log derivatives, now computing c_mat_xak')
    # compute c_mat_xak for each vertex in preparation for feasibility tests.
    for perm,vv in vertex_dict.items():
      vv.vertex_c_mat_xak_calc()
    print('Done.')

  def __get_dom_vec(self,a_mat,not_dominated_col_idx,dominated_row_vec_prev,dom_tuple_prev):
    perm_dict_add={} # use dictionary to make sure repeated ones are combined.
    for j in not_dominated_col_idx:
      dominated_row_j = a_mat[:,j]>0
      dominated_row_vec_new = dominated_row_j - dominated_row_j * dominated_row_vec_prev
      dominated_row_idx_new = np.where(dominated_row_vec_new)[0]
      dom_vec = np.array(dom_tuple_prev)
      dom_vec[dominated_row_idx_new] = j
      dominated_row_vec_now = dominated_row_vec_prev + dominated_row_vec_new
      if np.sum(dominated_row_vec_now)<a_mat.shape[0]:
        # there are rows not yet dominated
        not_dominated_row_idx_next = np.where(1-dominated_row_vec_now)[0]
        # then we combine all the rows not yet dominated to find the 
        #   columns not yet dominated. This could result in previously 
        #   already discovered dominance regimes to be counted again.
        not_dominated_col_idx_next = np.where(np.sum(a_mat[not_dominated_row_idx_next,:],axis=0)>0)[0]
        perm_dict_add_next = self.__get_dom_vec(a_mat,not_dominated_col_idx_next,dominated_row_vec_now,tuple(dom_vec))
        perm_dict_add.update(perm_dict_add_next)
      else: # all rows are dominated
        perm_dict_add[tuple(dom_vec)]=True
    return perm_dict_add

  def vertex_construct_direct(self):
    """
    Construct the rop_vertex objects that this binding network can have,
      directly, without feasibility test.

    Parameters
    ----------
    None.

    Returns
    -------
    None. The vertices are recorded in self.vertex_dict and by updating
      the vertex objects.
    """
    # We construct the vertices by iteratively construct all the possible
    # dominance vector (perm).
    print('Constructing vertex objects DIRECTLY...')
    a_mat=self.l_mat
    d,n=a_mat.shape
    not_dominated_col_idx=list(range(n))
    dominated_row_vec_prev=np.zeros(d)
    dom_tuple_prev=tuple(np.empty(d,dtype=int))
    perm_dict=self.__get_dom_vec(a_mat,not_dominated_col_idx,dominated_row_vec_prev,dom_tuple_prev)
    vertex_fin_dict={}
    vertex_inf_dict={}
    vertex_infHO_dict={}
    for perm in perm_dict.keys():
      vertex=rop_vertex(perm,self)
      repeat_vec=np.sum(vertex.p_mat,axis=0)
      if vertex.orientation!=0:
        vertex_fin_dict[perm]=vertex
      # elif np.max(repeat_vec)>2 or np.sum(repeat_vec>1) >=2:
        # there is an index repeated 3 or more times, OR there are more than two indices repeated twice or above;
        # So this is an infinite vertex of higher order.
      elif np.sum(repeat_vec[np.where(repeat_vec>1)[0]])>2:
        # another way to test for higher order is the sum of repeated indices' total repeated times is >2.
        vertex_infHO_dict[perm]=vertex
      else: #infinite vertex of order one
        vertex_inf_dict[perm]=vertex
    vertex_dict={**vertex_fin_dict,**vertex_inf_dict}
    self.vertex_dict={'all':vertex_dict,'finite':vertex_fin_dict,'infinite':vertex_inf_dict,'infiniteHO':vertex_infHO_dict}

    print('Finished vertex construction, now computing neighbors of vertices...')
    for perm,vv in vertex_dict.items():
      vv.vertex_find_neighbors()

    print('Finished neighbors, now computing log derivatives...')
    # first compute log der for finite vertices, since infinite ones rely on
    # finite neighbors to find orientation.
    for perm,vv in vertex_dict.items():
      vv.vertex_ld_calc()

    print('Finished log derivatives, now computing c_mat_xak')
    # compute c_mat_xak for each vertex in preparation for feasibility tests.
    for perm,vv in vertex_dict.items():
      vv.vertex_c_mat_xak_calc()
    print('Done.')

  def vertex_constrained_construct(self,opt_constraints,chart='xak'):
    """
    Assuming self.vertex_dict is already computed, for given 
      opt_constraints, this function computes whether the vertices
      are feasible under these constraints, update 
      rop_vertex.is_feasible for each vertex, and update each
      vertex's feasible neighbors (stored in 
      vertex.neighbors_constrined_dict) and return
      is_feasible_dict, a dictionary of {perm:is_feasible} pairs.
    This function calls vertex_list_feasibility_test.

    Parameters
    ----------
    opt_constraints : list of cvxpy inequalities
      List of constraints under which vertices are tested for feasibility.
    chart : str, optional
      A string with value from {'x','xak','tk'} that specifies the 
        chart that the opt_constraints are described in.

    Returns
    -------
    is_feasible_dict : dictionary
      A dictinoary of {perm:is_feasible} pairs for whether a vertex
        is feasible.
    """
    # for the given opt_constraints, test for each vertex whether it is feasible
    # under opt_constraints, and create a vertex dictionary for feasible vertices
    # under opt_constraints, stored in self.vertex_constrained_dict.
    # Also update each vertex's neighbors under constraints,
    # stored in each vertex.neighbors_constrained_dict.

    print('Compute feasible vertices...')
    for perm,vv in self.vertex_dict['all'].items():
      is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints)
      vv.is_feasible=is_feasible
    vertex_feasible_all={perm:vv for perm,vv in self.vertex_dict['all'].items() if vv.is_feasible}
    vertex_feasible_fin={perm:vv for perm,vv in self.vertex_dict['finite'].items() if vv.is_feasible}
    vertex_feasible_inf={perm:vv for perm,vv in self.vertex_dict['infinite'].items() if vv.is_feasible}
    self.vertex_constrained_dict={'all':vertex_feasible_all,'finite':vertex_feasible_fin,'infinite':vertex_feasible_inf}

    print('Compute neighbors under opt_constraints...')
    for perm,vv in self.vertex_dict['all'].items():
      if vv.is_feasible:
        # if it is feasible, we want to look at its neighbors.
        # if a neighbor is feasible, it is still a neighbor under constraint.
        # if a neighbor is infeasible, we look at its neighbors to see whether
        # they are feasible. This recurses.
        vv.vertex_update_constrained_neighbors()
    print('Done.')

  def vertex_list_feasibility_test(self,opt_constraints,chart='xak'):
    """
    Given opt_constraints, test all the vertices for their feasibility
      and return is_feasible_dict.
    This function is called by vertex_constrained_construct.
    It can also be directly called to test for feasibility without
      storing or finding feasible neighbors.

    Parameters
    ----------
    opt_constraints : list of cvxpy inequalities
      List of constraints under which vertices are tested for feasibility.
    chart : str, optional
      A string with value from {'x','xak','tk'} that specifies the 
        chart that the opt_constraints are described in.

    Returns
    -------
    is_feasible_dict : dictionary
      A dictinoary of {perm:is_feasible} pairs for whether a vertex
        is feasible.
    """
    # for the given opt_constraints, test each of the vertex whether it is feasible
    is_feasible_fin={}
    is_feasible_inf={}
    for perm,vv in self.vertex_dict['finite'].items():
      is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints)
      is_feasible_fin[perm]=is_feasible
    for perm,vv in self.vertex_dict['infinite'].items():
      is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints)
      is_feasible_inf[perm]=is_feasible

    is_feasible_all={**is_feasible_fin,**is_feasible_inf}
    is_feasible_dict={'all':is_feasible_all,'finite':is_feasible_fin,'infinite':is_feasible_inf}
    return is_feasible_dict

  def sampling_over_vertex_hull(self,nsample,vertex_perm_list=[],is_finite_only=False, chart='x',logmin=-6,logmax=6,margin=0,c_mat_extra=[],c0_vec_extra=[]):
    """
    Randomly sample points in the log space of chart variables,
      but instead of log-uniform, we first assign points to each vertex
      in an even fashion, then sample uniformly within each vertex.

    Parameters
    ----------
    nsample : int
      The number of points to be sampled in the space of chart variables.
      This is divided evenly to all the vertices of this binding network.
    is_finite_only : bool, optional
      Useful only when vertex_perm_list=[], so that all vertices are sampled.
      If True, only finite vertices are sampled. This also allows chart 'tk' to work.
      If False, both finite and infinite vertices are sampled.
      Defaults to False.
    vertex_perm_list : list of vertex's perm tuples, optional
      The list of perms indexing the vertices to be sampled.
      e.g. [(0,1,2),(0,1,3)].
      Defaults to empty list []. If empty, sample all vertices.
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'. Defaults to 'x'.
    margin : float, optional
      The vertex's feasibility conditions are inequalities, 
        of the form c_mat*logx + c0_vec > margin (e.g. in 'x' chart),
        where margin is the positive threshold used here. Default to 0.
      This can be adjusted to be stronger/weaker requirements on dominance.
    logmin : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    logmax : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.

    Returns
    -------
    sample_vertex_dict : dictionary of ndarray with shape nsample-by-dim_n
      Key is the perm of each vertex. Value is the sample for that vertex.
    """
    
    # calculate number of vertex to be plotted and the dictionary of vertices.
    vertex_plot_dict={}
    if vertex_perm_list: # vertex_perm_list is not empty
      nvertex=len(vertex_perm_list)
      for perm in vertex_perm_list:
        vertex_plot_dict[perm]=self.vertex_dict['all'][perm]
    else: # plot all vertices
      if is_finite_only: 
        finite_key='finite'
      else: 
        finite_key='all'
      nvertex=len(self.vertex_dict[finite_key].keys())
      vertex_plot_dict=self.vertex_dict[finite_key]
    # now sample each vertex.
    nsample_per_vertex=int(nsample/nvertex) # take the floor for number of sample per vertex
    sample_vertex_dict={}
    for key,vv in vertex_plot_dict.items():
      print(key)
      sample_vertex_dict[key]=vv.vertex_hull_sampling(nsample_per_vertex,chart=chart,margin=margin,logmin=logmin,logmax=logmax,c_mat_extra=c_mat_extra,c0_vec_extra=c0_vec_extra)
    return sample_vertex_dict

  def sampling_over_activity_regime_hull(self,nsample,b_tuple,regime_key_list=[],is_finite_only=False,is_feasible_only=False,chart='x',logmin=-6,logmax=6,margin=0,c_mat_extra=[],c0_vec_extra=[]):
    """
    Randomly sample points in the log space of chart variables for each
      dom_regime for a given activity.
      but instead of log-uniform, we first assign points to each vertex
      in an even fashion, then sample uniformly within each vertex.

    Parameters
    ----------
    nsample : int
      The number of points to be sampled in the space of chart variables.
      This is divided evenly to all the vertices of this binding network.
    b_tuple : tuple of length dim_n
      The b_tuple indicating the activity whose dom_regimes we are 
        interested in sampling.
    is_finite_only : bool, optional
      If True, only finite dom_regimes are sampled.
      If False, both finite and infinite dom_regimes are sampled.
      Defaults to False.
    regime_key_list : list of dominance regime's keys, optional
      The list of keys for dom_regimes indexing the dom_regimes 
        to be sampled. e.g. [((0,1,2),7),((0,1,3),7)].
      If empty, sample all dom_regimes.
      Defaults to empty list []. 
    is_feasible_only : bool, optional
      If True, only feasible dom_regimes are sampled.
      If False, all dom_regimes in regime_key_list (or all in this 
        activity) are sampled.
      Each dom_regime's is_feasible tag come from results of the most
        recent feasibility test.
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'. Defaults to 'x'.
    margin : float, optional
      The vertex's feasibility conditions are inequalities, 
        of the form c_mat*logx + c0_vec > margin (e.g. in 'x' chart),
        where margin is the positive threshold used here. Default to 0.
      This can be adjusted to be stronger/weaker requirements on dominance.
    logmin : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.
    logmax : float or ndarray vector
      logmin, logmax could be scalars, then it's the same value applied to 
        every variable. 
      They could also be vectors of length dim_n.

    Returns
    -------
    sample_vertex_dict : dictionary of ndarray with shape nsample-by-dim_n
      Key is the perm of each vertex. Value is the sample for that vertex.
    """
    pass
    # # calculate number of vertex to be plotted and the dictionary of vertices.
    # vertex_plot_dict={}
    # if vertex_perm_list: # vertex_perm_list is not empty
    #   nvertex=len(vertex_perm_list)
    #   for perm in vertex_perm_list:
    #     vertex_plot_dict[perm]=self.vertex_dict['all'][perm]
    # else: # plot all vertices
    #   if is_finite_only: 
    #     finite_key='finite'
    #   else: 
    #     finite_key='all'
    #   nvertex=len(self.vertex_dict[finite_key].keys())
    #   vertex_plot_dict=self.vertex_dict[finite_key]
    # # now sample each vertex.
    # nsample_per_vertex=int(nsample/nvertex) # take the floor for number of sample per vertex
    # sample_vertex_dict={}
    # for key,vv in vertex_plot_dict.items():
    #   sample_vertex_dict[key]=vv.vertex_hull_sampling(nsample_per_vertex,chart=chart,margin=margin,logmin=logmin,logmax=logmax)
    # return sample_vertex_dict

  def activity_regime_construct(self,b_vec):
    # given b_vec, go through all vertices and their possible regimes
    # test for basic feasibility, and return a dictionary of regimes
    # {(perm,row):regime}; perm is vertex, row is dominant species in activity (b_vec)
    # regime is an rop_dom_regime object.
    b_vec=np.array(b_vec)
    print('Computing feasible regimes...')
    nnz_b=np.where(b_vec>0)[0]
    regime_fin_dict={}
    regime_inf_dict={}
    for perm,vv in self.vertex_dict['all'].items():
      for j in nnz_b:
        regime=rop_dom_regime(j,b_vec,perm,self)
        # if vv.orientation == 0 and np.all(regime.ld==0): #this is an infinite vertex in direction not relevant for this activity
        #   continue
        is_feasible=regime.feasibility_test(chart='x',opt_constraints=[])
        if is_feasible:
          if vv.orientation == 0:
            regime_inf_dict[(perm,j)]=regime
          else: regime_fin_dict[(perm,j)]=regime

    regime_all_dict={**regime_fin_dict,**regime_inf_dict}
    self.activity_regime_dict[tuple(b_vec)]={'finite':regime_fin_dict,
                                            'infinite':regime_inf_dict,
                                            'all':regime_all_dict}

    print('Feasible regime computed, computing their neighboring regimes...')
    for key,regime in regime_all_dict.items():
      regime.find_neighbors()
    for key,regime in regime_all_dict.items():
      regime.find_neighbors_zero()

    print('Computing activity logder regimes dictionary... ')
    # since several regimes will have the same log derivative, we use log derivative as key, rather than regimes.
    ld_regime_fin_key_dict={}
    ld_regime_inf_key_dict={}
    for key,regime in regime_fin_dict.items():
      try: ld_regime_fin_key_dict[tuple(regime.ld)]+=[key] #add regime to ld
      except KeyError: ld_regime_fin_key_dict[tuple(regime.ld)]=[key] #no regimes yet, initiate.
    for key,regime in regime_inf_dict.items():
      try: ld_regime_inf_key_dict[tuple(regime.ld)]+=[key] #add regime to ld
      except KeyError: ld_regime_inf_key_dict[tuple(regime.ld)]=[key] #no regimes yet, initiate.
    # convert into dictionary of ld_regime objects
    ld_regime_fin_dict={}
    ld_regime_inf_dict={}
    is_ray=False
    for ld,regime_keys in ld_regime_fin_key_dict.items():
      ld_regime=rop_ld_regime(ld,is_ray,b_vec,regime_keys,self)
      ld_regime_fin_dict[ld]=ld_regime
    is_ray=True
    for ld,regime_keys in ld_regime_inf_key_dict.items():
      ld_regime=rop_ld_regime(ld,is_ray,b_vec,regime_keys,self)
      ld_regime_inf_dict[ld]=ld_regime

    ld_regime_all_dict={**ld_regime_fin_dict,**ld_regime_inf_dict}
    self.activity_ld_regime_dict[tuple(b_vec)]={'finite':ld_regime_fin_dict,
                                               'infinite':ld_regime_inf_dict,
                                               'all':ld_regime_all_dict}

    print('Computing activity logder regimes neighbors... ')
    for key,ld_regime in ld_regime_all_dict.items():
      ld_regime.find_neighbors()

    print('Done')

  def activity_regime_constrained_construct(self,b_vec,opt_constraints,chart='xak'):
    # For the given opt_constraints, test for each dom_regime whether it is
    # feasible, update their neighbors under constraint,
    # and create a dictionary for feasible dom_regimes, stored in
    # self.activity_regime_contrained_dict[b_vec].
    # and create a dictionary for feasible ld_regimes, stored in
    # self.activity_ld_regime_constrained_dict[b_vec].
    print("Compute dominance regimes' feasibility...")
    regime_constrained_fin={}
    regime_constrained_inf={}
    for key,regime in self.activity_regime_dict[tuple(b_vec)]['all'].items():
      is_feasible=regime.feasibility_test(chart=chart,opt_constraints=opt_constraints)
      regime.is_feasible=is_feasible
      if is_feasible:
        if regime.vertex.orientation==0:
          regime_constrained_inf[key]=regime
        else:
          regime_constrained_fin[key]=regime
    regime_constrained_all={**regime_constrained_fin,**regime_constrained_inf}
    self.activity_regime_constrained_dict[tuple(b_vec)]={'all':regime_constrained_all,'finite':regime_constrained_fin,'infinite':regime_constrained_inf}

    print("Compute regimes' neighboring regimes under opt_constraints...")
    for key,regime in self.activity_regime_dict[tuple(b_vec)]['all'].items():
      if regime.is_feasible:
        # if it is feasible, we want to look at its neighbors.
        # if a neighbor is feasible, it is still a neighbor.
        # if a neighbor is infeasible, we look at its neighbors to see whether
        # they are feasible.
        regime.update_constrained_neighbors()

    print("Compute ld regimes' feasibility and compute neighbors under opt_constraints...")
    ld_regime_constrained_fin={}
    ld_regime_constrained_inf={}
    for ld,ld_regime in self.activity_ld_regime_dict[tuple(b_vec)]['all'].items():
      ld_regime.update_feasibility()
      if ld_regime.is_feasible:
        ld_regime.update_constrained_neighbors()
        if ld_regime.is_ray:
          ld_regime_constrained_inf[ld]=ld_regime
        else:
          ld_regime_constrained_fin[ld]=ld_regime
    ld_regime_constrained_all={**ld_regime_constrained_fin,**ld_regime_constrained_inf}
    self.activity_ld_regime_constrained_dict[tuple(b_vec)]={
        'all':ld_regime_constrained_all,
        'finite':ld_regime_constrained_fin,
        'infinite':ld_regime_constrained_inf
        }

    print('Done')

  def activity_list_feasibility_test(self,b_vec,opt_constraints,chart='xak'):
    # for the activity defined by b_vec (b_vec*x), and constraints given in
    # opt_constraints in chart (default xak), test each of the ld_regime
    # whether it is feasible.
    # A ld_regime is feasible if one of its dom_regime is feasible.
    is_feasible_fin={}
    is_feasible_inf={}
    for ld,ld_regime in self.activity_ld_regime_dict[tuple(b_vec)]['finite'].items():
      is_feasible_dom_regime_list=[dom_regime.feasibility_test(chart=chart,opt_constraints=opt_constraints) for dom_regime in ld_regime.dom_regime_dict.values()]
      is_feasible=np.any(is_feasible_dom_regime_list)
      is_feasible_fin[ld]=is_feasible
    for ld,ld_regime in self.activity_ld_regime_dict[tuple(b_vec)]['infinite'].items():
      is_feasible_dom_regime_list=[dom_regime.feasibility_test(chart=chart,opt_constraints=opt_constraints) for dom_regime in ld_regime.dom_regime_dict.values()]
      is_feasible=np.any(is_feasible_dom_regime_list)
      is_feasible_inf[ld]=is_feasible
    is_feasible_all={**is_feasible_fin,**is_feasible_inf}
    is_feasible_dict={'all':is_feasible_all,'finite':is_feasible_fin,'infinite':is_feasible_inf}
    return is_feasible_dict

# BELOW ARE SYMBOLIC METHODS for the binding network

  def logder_sym(self,a_mat=np.array([]),is_saved=True):
    """calculate the symbolic log derivative matrix of dlog(x)/dlog(a_mat*x,k).

    Parameters
    ----------
    a_mat : numpy array, optional
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.
      Optional, defaults to l_mat of the binding network.
    is_saved: 'bool', optional
      To save the resulting logdermat as parameter ld_sym of the binding network or not.
      Defaults to True.
    Returns
    -------
    logdermat: sympy symbolic matrix
       Symbolic log derivative matrix for x to (t_sym_temp, k_sym) expressed in terms of x.

    Note that if the provided a_mat yields a noninvertible matrix, then the function will
    return logdermat as a matrix of zeros (with the appropriate size).
    """
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    else:
      assert a_mat.shape==(self.dim_d,self.dim_n), f"the shape of L matrix should be {self.dim_d} by {self.dim_n}."
      assert np.all(a_mat>=0), "all entries of A matrix should be non-negative."
      assert np.all(a_mat.dot(np.ones(self.dim_n))>0), "each row of A matrix should have at least one positive entry."

    a_mat_sym=sp.Matrix(a_mat)
    n_mat_sym=self.n_mat_sym

    #it's much faster (10x) to use a dummy "total" variables to calculate matrix inverse,
    # and then substitute the expression for the totals in terms of x, rather than
    # inverting with the totals explicitly expressed in x variables.

    temp_sym = sp.symbols("t:"+str(self.dim_d))

    Lam_t = sp.Matrix.diag(*temp_sym)
    Lam_x = sp.Matrix.diag(*self.x_sym)

    topmat = Lam_t**-1 * a_mat_sym * Lam_x
    fullmat = topmat.col_join(n_mat_sym)

    logdermat_t = fullmat**-1

    temp_expr=a_mat_sym*sp.Matrix(self.x_sym)
    temp2x_subs_list=[(temp_sym[i],temp_expr[i]) for i in range(self.dim_d)]
    logdermat=sp.simplify(logdermat_t.subs(temp2x_subs_list))

    if is_saved: self.ld_sym_dict[a_mat.tobytes()]=logdermat
    return logdermat

  def subs_list_t2x(self):
    total_expr=self.l_mat_sym*sp.Matrix(self.x_sym)
    t2x_subs_list=[(self.t_sym[i],total_expr[i]) for i in range(self.dim_d)]
    return t2x_subs_list

  def subs_list_xc2xak(self):
    assert self.is_atomic, "this operation requires the binding network to be atomic"
    l2_mat_sym=self.l_mat_sym[:,self.dim_d:]
    n2_mat_sym=self.n_mat_sym[:,self.dim_d:]
    xa_sym_vec=sp.Matrix(self.x_sym[:self.dim_d])
    xa_sym_log=xa_sym_vec.applyfunc(sp.log)
    xc2xak_log_expr=l2_mat_sym.T*xa_sym_log+n2_mat_sym.inv()*sp.Matrix(self.k_sym).applyfunc(sp.log)
    func_exponentiate=lambda x:sp.E**x
    xc2xak_expr=xc2xak_log_expr.applyfunc(func_exponentiate)
    xc2xak_subs_list=[(self.x_sym[self.dim_d+i],xc2xak_expr[i]) for i in range(self.dim_r)]
    return xc2xak_subs_list

  def t2x_sym(self,expr):
    """input a symbolic expression containing totals t, map it to x

    Parameters
    ----------
    expr : sympy symbolic expression
       A symbolic expression to be converted

    Returns
    -------
    expr_x: sympy symbolic expression
       The symbolic expression after conversion.
    """
    # calculate the substitutions map for totals to species
    try: expr_x=expr.subs(self.t2x)
    except AttributeError:
      t2x_subs_list=self.subs_list_t2x()
      self.t2x=t2x_subs_list
      expr_x=expr.subs(self.t2x)
    return expr_x

  def xc2xak_sym(self,expr):
    """input a symbolic expression containing complex species x^c, map it to x^a and k.

    Parameters
    ----------
    expr : sympy symbolic expression
       A symbolic expression to be converted

    Returns
    -------
    expr_xak: sympy symbolic expression
       The symbolic expression after conversion.
    """
    assert self.is_atomic, "this operation requires the binding network to be atomic"
    # calculate the substitutions map for complex species to atomic species and k's, if it does not exist.
    try: expr_xak=expr.subs(self.xc2xak)
    except AttributeError:
      xc2xak_subs_list=self.subs_list_xc2xak()
      self.xc2xak=xc2xak_subs_list
      expr_xak=expr.subs(self.xc2xak)
    return expr_xak

  def logder_sym_activity(self,b_vec_sym,a_mat=np.array([])):
    """compute the log derivative of a linear sum of x, i.e. dlog(b_vec_sym*x)/dlog(tp,k), tp=a_mat*x

    Parameters
    ----------
    b_vec_sym : sympy symbolic vector, shape n-by-1
      A vector of symbolic expressions corresponding to coefficients to be summed
    a_mat: numpy array, optional
      The matrix denoting the variables a_mat*x that the log derivative is taken with respect to.
      Defaults to self.l_mat.

    Returns
    -------
    ld_sym_sum: sympy vector of symbolic expression
       The symbolic log derivative  dlog(b_vec_sym*x)/dlog(tp,k), tp=a_mat*x.
    """
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    # check if logder is already calculated with respect to this a_mat.
    try: ld_sym=self.ld_sym_dict[a_mat.tobytes()]
    except KeyError: # not calculated yet
      print('Symbolic logder for this chart coordinate does not exist, calculating...')
      ld_sym=self.logder_sym(a_mat=a_mat)
    coeff_vec_sym=sp.Matrix(np.zeros(b_vec_sym.shape))
    bx = sp.matrix_multiply_elementwise(b_vec_sym,sp.Matrix(self.x_sym)) #b multiply x element-wise
    bx_sum = bx.T*sp.ones(*bx.shape)
    coeff_vec_sym=bx*bx_sum**(-1) # convex coefficients for the log derivative sum
    ld_sym_sum=(ld_sym.T*coeff_vec_sym).T # for logder matrix ld_sym, rows are species, columns are variables.
    return ld_sym_sum