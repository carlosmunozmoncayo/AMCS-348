#Some Optimal SSP RK schemes
#Taken from Gottlieb, Ketcheson, and Shu (2011)

def SSPRK22(fun, u0, dt, t0, tfinal, nframes):
    n = len(u0)
    t = t0
    u = np.copy(u0)
    u_frames = [u0]
    t_eval = [t0]

    #For sampling solution (needs improvement)
    n_outputs = int((tfinal-t0)/dt)
    skip = n_outputs//nframes
    counter_skip = 0

    while t<tfinal:
        u1 = u + dt*fun(u)
        u  = 0.5*u + 0.5* u1 +0.5*dt*fun(u1)
        
        #Saving solution
        if counter_skip > skip:
            u_frames.append(np.copy(u))
            t_eval.append(t)
            counter_skip=0
        else:
            counter_skip += 1
        
        t += dt
    return u_frames, t_eval

def SSPRK33(fun, u0, dt, t0, tfinal, nframes):
    n = len(u0)
    t = t0
    u = np.copy(u0)
    u_frames = [u0]
    t_eval = [t0]

    #For sampling solution (needs improvement)
    n_outputs = int((tfinal-t0)/dt)
    skip = n_outputs//nframes
    counter_skip = 0

    while t<tfinal:
        u1 = u + dt*fun(u)
        u2 = 0.75*u +0.25*u1 + 0.25*dt*fun(u1)
        u  = (1./3.)*u + (2./3.)* u2 +(2./3.)*dt*fun(u2)
        
        #Saving solution
        if counter_skip > skip:
            u_frames.append(np.copy(u))
            t_eval.append(t)
            counter_skip=0
        else:
            counter_skip += 1
        
        t += dt
    return u_frames, t_eval


def SSPRK53(fun, u0, dt, t0, tfinal, nframes):
    n = len(u0)
    t = t0
    u = np.copy(u0)
    u_frames = [u0]
    t_eval = [t0]

    #For sampling solution (needs improvement)
    n_outputs = int((tfinal-t0)/dt)
    skip = n_outputs//nframes
    counter_skip = 0

    while t<tfinal:
        u1 = u + 0.37726891511710*dt*fun(u)
        u2 = u1 + 0.37726891511710*dt*fun(u1)
        u3 = (0.56656131914033*u+ 0.43343868085967*u2+
              0.16352294089771*dt*fun(u2))
        u4= (0.09299483444413*u+ 0.00002090369620*u1+
             0.90698426185967*u3+ 0.00071997378654*dt*fun(u)+
             0.34217696850008*dt*fun(u3))
        u  = (0.00736132260920*u+ 0.20127980325145*u1+
              0.00182955389682*u2+ 0.78952932024253*u4+
              0.00277719819460*dt*fun(u) + 0.00001567934613*dt*fun(u1)+
              0.29786487010104*dt*fun(u1))
        #Saving solution
        if counter_skip > skip:
            u_frames.append(np.copy(u))
            t_eval.append(t)
            counter_skip=0
        else:
            counter_skip += 1
        
        t += dt
    return u_frames, t_eval

def SSPRK54(fun, u0, dt, t0, tfinal, nframes):
    n = len(u0)
    t = t0
    u = np.copy(u0)
    u_frames = [u0]
    t_eval = [t0]

    #For sampling solution (needs improvement)
    n_outputs = int((tfinal-t0)/dt)
    skip = n_outputs//nframes
    counter_skip = 0

    while t<tfinal:
        u1 = u + 0.39175222657189*dt*fun(u)
        u2 = (0.444370493651235*u +0.555629506348765*u1+
              0.36841059305037*dt*fun(u1))
        u3 = (0.620101851488403*u + 0.379898148511597*u2+
              0.251891774271694*dt*fun(u2))
        u4 = (0.178079954393132*u + 0.821920045606868*u3+
              0.544974750228521*dt*fun(u3))
        u = (0.517231671970585*u2 + 0.096059710526147*u3+
             0.063692468666290*dt*fun(u3) + 0.386708617503269*u4+
             0.226007483236906*dt*fun(u4))

        #Saving solution
        if counter_skip > skip:
            u_frames.append(np.copy(u))
            t_eval.append(t)
            counter_skip=0
        else:
            counter_skip += 1
        
        t += dt
    return u_frames, t_eval
