import numpy as np
import pickle
import py3nj

ellmax = 1000

# #load wigner3j symbols
# wigner_file = '/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_ellmax1000.p'
# wigner_nonzero_m_file = '/global/homes/k/kmsurrao/NILC-Parameter-Pipeline/wigner3j_nonzero_m_ellmax1000.p'
# wigner_zero_m = pickle.load(open(wigner_file, 'rb'))[:ellmax+1, :ellmax+1, :ellmax+1]
# wigner_nonzero_m = pickle.load(open(wigner_nonzero_m_file, 'rb'))[:ellmax+1, :ellmax+1, :]
# print('loaded wigner-3j', flush=True)


#dummy power spectra
comp_cl = np.ones(ellmax+1)
M = np.ones(ellmax+1)
W = np.ones(ellmax+1)

#calculate spectra of masked map from MASTER approach
# l2 = np.arange(ellmax+1)
# l3 = np.arange(ellmax+1)
# m_array = np.zeros((ellmax+1,2001)) #acount for (-1)^{m_2+m_3} factor in term3
# zero_idx = 1000
# for l in range(ellmax+1):
#     for i in range(zero_idx-l, zero_idx+l+1):
#         if abs(i-zero_idx)%2==1:
#             m_array[l][i] = -1
#         else:
#             m_array[l][i] = 1
# print('filled in m_array', flush=True)
# term1 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,comp_cl,M,optimize=True)
# term2 = float(1/(4*np.pi))*np.einsum('a,b,lab,lab,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,W,W,optimize=True)
# term3 = float(1/(4*np.pi))*np.einsum('a,b,laa,lbb,lac,lbd,a,b,ac,bd->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,wigner_nonzero_m,wigner_nonzero_m,W,W,m_array,m_array,optimize=True)
# term3_test = float(1/(4*np.pi))*np.einsum('a,b,laa,lbb,lac,lbd,a,b,ac,bd,a,b->l',2*l2+1,2*l3+1,wigner_zero_m,wigner_zero_m,wigner_nonzero_m,wigner_nonzero_m,W,W,m_array,m_array,l2_test,l3_test,optimize=True)
# print('term1: ', term1[:30], flush=True)
# print('term2: ', term2[:30], flush=True)
# master_cl = term1 + term2 

# #test sum of three 3j -> 6j simplification of bispectrum terms
# ellmin = 0
# ellmax= 4
# for l1 in range(ellmin,ellmax):
#     for l2 in range(ellmin,ellmax):
#         for l3 in range(ellmin,ellmax):
#             for l4 in range(ellmin,ellmax):
#                 for l5 in range(ellmin,ellmax):
#                     tot_before = 0 #sum over m before simplification
#                     tot_after = 0 #sum after simplification
#                     for m3 in range(-l3, l3+1):
#                         for m4 in range(-l4, l4+1):
#                             term_after = (-1)**(l1+l2+l5)*py3nj.wigner3j(2*l3,2*l3,2*l4,2*m3,2*m3,2*m4)*py3nj.wigner6j(2*l3,2*l3,2*l4,2*l1,2*l2,2*l5)
#                             tot_after += term_after
#                             for m1 in range(-l1, l1+1):
#                                 for m2 in range(-l2, l2+1):
#                                     for m5 in range(-l5, l5+1):
#                                         term_before = (-1)**(m2+m3)*py3nj.wigner3j(2*l2,2*l3,2*l5,2*m2,2*m3,2*m5)*py3nj.wigner3j(2*l1,2*l3,2*l5,2*m1,-2*m3,2*m5)*py3nj.wigner3j(2*l1,2*l2,2*l4,2*m1,-2*m2,2*m4)
#                                         tot_before += term_before
#                     if abs(tot_before) != abs(tot_after):
#                         print(f'{tot_before}, {tot_after}, {l1}, {l2}, {l3}, {l4}, {l5}')

# #test whether (l1 l3 l5 0 0 0) (l2 l3 l5 0 0 0) is (l1 l3 l5 0 0 0)^2 -- it's not
# ellmax = 10
# for l1 in range(ellmax):
#     for l2 in range(ellmax):
#         for l3 in range(ellmax):
#             for l5 in range(ellmax):
#                 before = py3nj.wigner3j(2*l1,2*l3,2*l5,0,0,0)*py3nj.wigner3j(2*l2,2*l3,2*l5,0,0,0)
#                 after = py3nj.wigner3j(2*l1,2*l3,2*l5,0,0,0)**2
#                 if before != after:
#                     print(f'{before}, {after}, {l1}, {l2}, {l3}, {l5}')


#test whether (l3, l3, l4, m3, m3, m4) is zero
ellmax = 5
for l3 in range(ellmax):
    for l4 in range(ellmax):
        for m3 in range(-l3,l3+1):
            for m4 in range(-l4,l4+1):
                if l4==m4==0:
                    before = py3nj.wigner3j(2*l3,2*l3,2*l4,2*m3,2*m3,2*m4)
                    if before != 0:
                        print(f'{before}, {l3}, {l4}, {m3}, {m4}')


