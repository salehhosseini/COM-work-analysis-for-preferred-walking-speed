import numpy             as np
import matplotlib.pyplot as plt

## Initial Conditions
alpha = 0.35
v_ave = 1.25     # m/s

''' Section One: Even Walking ----------------------------------------------'''
### Optimum push-off for even walking for the given average walking velocity

# from Kuo 2002: velocity before step-to-step transition is v_ave
po_opt  = v_ave * np.tan(alpha); PO_opt = 0.5 * po_opt**2

# applying a random push-off
po_rand = np.linspace(0, po_opt, 100); 
PO_rand = 0.5 * po_rand**2

# velocity trajectory change in the middle
gama = np.arctan(po_rand / v_ave)

# Required collision
co_rand = po_rand * np.sin(2* alpha - gama) / np.sin(gama)
CO_rand = 0.5 * co_rand**2

## if the compensatoin is provided by the push-off (Ankle joint along the trailing leg)
v_aft_partial = v_ave * np.cos(2 * alpha - gama) / np.cos(gama)
compensation  = 0.5 * (v_ave**2 - v_aft_partial**2)
compensation_impulse = np.sqrt(2 * compensation)

percent = np.linspace(0,100,100)

#### when the push-off is more than the nominal push-off
po_max = v_ave * np.tan(2 * alpha); 
po_exc = np.linspace(po_opt, po_max, 100)
PO_exc = 0.5 * po_exc**2

gama_exc = np.arctan(po_exc / v_ave)
co_exc = po_exc * np.sin(2* alpha - gama_exc) / np.sin(gama_exc)
CO_exc = 0.5 * co_exc**2

v_post_exc = v_ave * np.cos(2 * alpha - gama_exc) / np.cos(gama_exc)
post_dissipation = 0.5 * v_post_exc**2 - 0.5 * v_ave**2

percent_exc = np.linspace(100, (po_max / po_opt) * 100, 100)

''' Section Two: Uneven Walking --------------------------------------------'''
# Calculating the optimum push-off and collision for a given velocity when the 
# walker encounters a step-up: dh

## The push-off must mechanically energize the step-to-step transition and work against
## the gravity

g  = 9.81    # gravitational acceleration
dh = 0.025   # up-step

# optimal velocity at the beginning of the upstep stance
v_aft_opt_uneven = np.sqrt(v_ave**2 + 2 * g * dh)
## optimum angle of mid transition velocity
gama_star_uneven = np.arctan((1/np.sin(2 * alpha)) * (v_aft_opt_uneven / v_ave - np.cos(2 * alpha)))
## optimum push-off impulse and its work
po_opt_uneven = v_ave * np.tan(gama_star_uneven); PO_opt_uneven = 0.5 * po_opt_uneven**2
## optimum collision
co_opt_uneven = po_opt_uneven * np.sin(2 * alpha - gama_star_uneven) / np.sin(gama_star_uneven)
CO_opt_uneven = 0.5 * co_opt_uneven**2

# A random push-off and its work        
po_rand_uneven = np.linspace(0, po_opt_uneven, 100); PO_rand_uneven = 0.5 * po_rand_uneven**2
# the resulted agle of mid-tranisiton velocity
gama_rand_uneven = np.arctan(po_rand_uneven / v_ave)
# the resulted collision
co_rand_uneven = v_ave * np.sin(2 * alpha - gama_rand_uneven) / np.cos(gama_rand_uneven)
## Work of collision
CO_rand_uneven = 0.5 * co_rand_uneven**2
## partial final velocity
v_aft_partial_uneven = v_ave * np.cos(2 * alpha - gama_rand_uneven) / np.cos(gama_rand_uneven)
# The required compensation (impulse):
compensation = 0.5 * (v_aft_opt_uneven**2 - v_aft_partial_uneven**2)
compensation_impulse = np.sqrt(2 * compensation)

## when the push-off is more than the optimal push-off
po_exc_uneven = po_exc[po_exc > po_opt_uneven]
PO_exc_uneven = 0.5 * po_exc_uneven**2

co_exc_uneven = co_exc[co_exc < co_opt_uneven]
CO_exc_uneven = 0.5 * co_exc_uneven**2

v_post_exc_uneven = v_post_exc[v_post_exc > v_aft_opt_uneven]
post_dissipation_uneven = 0.5 * v_post_exc_uneven**2 - 0.5 * v_ave**2 - g * dh

percent_exc_uneven = np.linspace(100, (po_max / po_opt_uneven) * 100, len(PO_exc_uneven))

## total dissipations
diss_sub_even = CO_rand / np.max(PO_rand)   ## sub-optimal push-off
diss_exc_even = CO_exc / np.max(PO_rand) + (PO_exc - CO_exc) / np.max(PO_rand)  ## excess puhs-off
diss_even = np.append(diss_sub_even, diss_exc_even)

diss_sub_uneven = CO_rand_uneven / PO_opt_uneven  ## sub-optimal push-off
diss_exc_uneven = CO_exc_uneven / PO_opt_uneven + (PO_exc_uneven - CO_exc_uneven - g * dh) / PO_opt_uneven
# diss_exc_uneven = (PO_exc_uneven - g * dh) / PO_opt_uneven
diss_uneven = np.append(diss_sub_uneven, diss_exc_uneven)


''' Section three: plotting ------------------------------------------------''' 
plt.figure(1, figsize = (14, 6))

plt.subplot(2,2,1); 
plt.plot(percent, PO_rand / np.max(PO_rand), color = 'darkred', linewidth =3, label = 'Push-off')
plt.plot(percent, CO_rand / np.max(PO_rand), color = 'darkblue', linewidth =3, label = 'Collision', linestyle = '--')

plt.title('A. Even walking push-off and collision profiles', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize =12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize =12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

#
plt.subplot(2,2,3);
plt.plot(percent, (CO_rand - PO_rand) / np.max(PO_rand), color = 'darkred', linewidth =3, label = 'Post transition compensation')
plt.plot(percent[79], ((CO_rand - PO_rand) / np.max(PO_rand))[79], '*r')
plt.plot(percent, CO_rand / np.max(PO_rand), color = 'darkgreen', linewidth =3, label = 'Total positive work', linestyle = '--')
plt.plot(percent[79], (CO_rand / np.max(PO_rand))[79], '*r')

plt.title('C. Even walking post transition compensation and total positive work profiles', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize =12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize =12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

print('post transition deficit (even): ', ((CO_rand - PO_rand) / np.max(PO_rand))[79])
print('total work (even): ',(CO_rand / np.max(PO_rand))[79])

#
plt.subplot(2,2,2)
plt.plot(percent, PO_rand_uneven / PO_opt_uneven, color = 'darkred', linewidth =3, label = 'Push-off')
plt.plot(percent, CO_rand_uneven / PO_opt_uneven, color = 'darkblue', linewidth =3, label = 'Collision', linestyle = '--')

plt.title('B. Uneven walking push-off and collision profiles', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize = 12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize = 12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

#
plt.subplot(2,2,4)
plt.plot(percent, (CO_rand_uneven - PO_rand_uneven + g * dh) / PO_opt_uneven, color = 'darkred', linewidth =3, label = 'Post transition compensation')
plt.plot(percent[79], ((CO_rand_uneven - PO_rand_uneven + g * dh) / PO_opt_uneven)[79], '*r')
plt.plot(percent, (CO_rand_uneven + g * dh) / PO_opt_uneven, color = 'darkgreen', linewidth =3, label = 'Total positive work', linestyle = '--')
plt.plot(percent[79], ((CO_rand_uneven + g * dh) / PO_opt_uneven)[79], '*r')

plt.title('D. Uneven walking post transition compensation and total positive work profiles', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize = 12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize = 12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

plt.tight_layout()

##
po_traj = PO_rand_uneven / PO_opt_uneven
co_traj = CO_rand_uneven / PO_opt_uneven

print('gravity work: ', (po_traj - co_traj)[-1])

print('post transition deficit (uneven): ', ((CO_rand_uneven - PO_rand_uneven + g * dh) / PO_opt_uneven)[79])
print('total work (uneven):', ((CO_rand_uneven + g * dh) / PO_opt_uneven)[79])

####
plt.figure(2, figsize = (14, 6))
plt.subplot(2,2,1)
plt.plot(percent_exc, PO_exc / np.max(PO_rand), color = 'darkred', linewidth =3, label = 'Push-off')
plt.plot(percent_exc, CO_exc / np.max(PO_rand), color = 'darkblue', linewidth =3, label = 'Collision')

plt.plot(percent_exc[19], (PO_exc / np.max(PO_rand))[19], '*r')
plt.plot(percent_exc[19], (CO_exc / np.max(PO_rand))[19], '*r')

plt.title('A. Even walking push-off and collision profiles', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize =12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize =12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

plt.subplot(2,2,3)
plt.plot(percent_exc, (PO_exc - CO_exc) / np.max(PO_rand), color = 'darkred', linewidth =3, label = 'Excess mechanical energy')
plt.plot(percent_exc, post_dissipation / np.max(PO_rand), color = 'darkblue', linewidth =2, label = 'Required mechanical energy dissipation', linestyle = '--')

plt.plot(percent_exc[19], ((PO_exc - CO_exc) / np.max(PO_rand))[19], '*r')
plt.plot(percent_exc[19], (post_dissipation / np.max(PO_rand))[19], '*r')
print('even walking excess energy dissipation:', ((PO_exc - CO_exc) / np.max(PO_rand))[19])
print('even walking excess push-off impulse: ', percent_exc[19])

plt.title('C. Even walking post transition required energy dissipation', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize =12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize =12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

plt.subplot(2,2,2)
plt.plot(percent_exc_uneven, PO_exc_uneven / PO_opt_uneven, color = 'darkred', linewidth =3, label = 'Push-off')
plt.plot(percent_exc_uneven, CO_exc_uneven / PO_opt_uneven, color = 'darkblue', linewidth =3, label = 'Collision')

plt.plot(percent_exc_uneven[30], (PO_exc_uneven / PO_opt_uneven)[30], '*r')
plt.plot(percent_exc_uneven[30], (CO_exc_uneven / PO_opt_uneven)[30], '*r')

plt.title('B. Uneven walking push-off and collision profiles', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize = 12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize = 12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

plt.subplot(2,2,4)
plt.plot(percent_exc_uneven, (PO_exc_uneven - CO_exc_uneven - g * dh) / PO_opt_uneven, color = 'darkred', linewidth =3, label = 'Excess mechanical energy')
plt.plot(percent_exc_uneven, post_dissipation_uneven / PO_opt_uneven, color = 'darkblue', linewidth =2, label = 'Required mechanical energy dissipation', linestyle = '--')

plt.plot(percent_exc_uneven[30], ((PO_exc_uneven - CO_exc_uneven - g * dh) /PO_opt_uneven)[30], '*r')
plt.plot(percent_exc_uneven[30], (post_dissipation_uneven / PO_opt_uneven)[30], '*r')

plt.title('D. Uneven walking post transition required energy dissipation', loc = 'left', fontsize = 12.5)
plt.xlabel('% Nominal Push-off Impulse', fontsize = 12.5)
plt.ylabel('Optimal Push-Off (J/J)', fontsize = 12.5)
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)
plt.legend(frameon = False)

print('Uneven walking excess push-off dissipation: ', (post_dissipation_uneven / np.max(PO_rand))[30])

plt.tight_layout()

###
plt.figure(3, figsize = (14, 6))
plt.subplot(1,2,1)
plt.plot(diss_even, color = 'navy', linewidth = 2.5, label = 'Total Step Dissipation')
plt.plot(np.append(np.full(100, np.nan), CO_exc / np.max(PO_rand)), linewidth = 2.5, linestyle = '--', color = 'darkgrey', label = 'Collision Dissipation')
plt.plot(np.append(np.full(100, np.nan), (PO_exc - CO_exc) / np.max(PO_rand)), linewidth = 2.5, linestyle = '--', 
         color = 'k', label = 'Post Transition Dissipation')

plt.legend(frameon = False)
plt.xlabel('% Nominal Push-off Impulse', fontsize = 12.5)
plt.ylabel('Total Step Dissipation (J/J)', fontsize = 12.5)
plt.title('A. Even Walking', fontsize = 12.5, loc = 'left')
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)

plt.subplot(1,2,2)
plt.plot(diss_uneven, color = 'navy', linewidth = 2.5, label = 'Total Step Dissipation')
plt.plot(np.append(np.full(100, np.nan), CO_exc_uneven / PO_opt_uneven), linewidth = 2.5, linestyle = '--', color = 'darkgrey', label = 'Collision Dissipation')
plt.plot(np.append(np.full(100, np.nan), (PO_exc_uneven - CO_exc_uneven - g * dh) / PO_opt_uneven), linewidth = 2.5, linestyle = '--', 
         color = 'k', label = 'Post Transition Dissipation')

plt.legend(frameon = False)
plt.xlabel('% Nominal Push-off Impulse', fontsize = 12.5)
plt.ylabel('Total Step Dissipation (J/J)', fontsize = 12.5)
plt.title('B. Uneven Walking', fontsize = 12.5, loc = 'left')
plt.xticks(fontsize = 12.5)
plt.yticks(fontsize = 12.5)

plt.tight_layout()

