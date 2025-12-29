#mcandrew

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam
from functools import partial

class puca( object ):

    def __init__(self
                 , y                 = None
                 , target_indicators = None
                 , X                 = None):

        self.X__input          = X
        self.y__input          = y
        self.target_indicators = target_indicators
        
        self.organize_data()

    @staticmethod
    def smooth_gaussian_anchored(x, sigma=2.0):
        """
        Heavy 1D/2D smoothing with a Gaussian kernel.
        - Uses reflect padding to avoid edge artifacts.
        - Forces first/last value of the smoothed series to equal the original.
        - Optimized to handle 2D arrays (smooths along axis 0)
        """
        x = np.asarray(x, float)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.reshape(-1, 1)
        
        radius = int(3 * sigma)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        kernel /= kernel.sum()

        # Optimization 3: Vectorize smoothing for 2D arrays
        # Process all columns at once
        x_pad = np.pad(x, pad_width=((radius, radius), (0, 0)), mode="reflect")
        
        # Apply convolution to each column
        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            y_full = np.convolve(x_pad[:, i], kernel, mode="same")
            y[:, i] = y_full[radius:-radius]

        # anchor endpoints
        y[0, :] = x[0, :]
        y[-1, :] = x[-1, :]
        
        return y.ravel() if is_1d else y
       
    def organize_data(self):

        y_input           = self.y__input
        target_indicators = self.target_indicators
        X                 = self.X__input

        #--split y data into examples from the past and the targets
        Y,y                 = zip(*[ (np.delete(_,t,axis=1), _[:,t])  for t,_ in zip(target_indicators, y_input)])

        #--This block standardizes Y data to z-scores, collects the mean and sd, and smooths past Ys.
        all_y             = np.array([])
        smooth_ys         = [] 
        y_means, y_scales = [] , []
        for n,(past,current) in enumerate(zip(Y,y)):
            means  = np.mean(past,axis=0)
            scales =  np.std(past,axis=0)

            y_means.append(means)
            y_scales.append(scales)

            past_smooth       =  self.smooth_gaussian_anchored(past,2)
            past_smooth_means =  np.mean(past_smooth,axis=0)
            past_smooth_stds  =  np.std(past_smooth,axis=0)

            smooth_y          = (past_smooth - past_smooth_means)/past_smooth_stds
            smooth_ys.append(smooth_y)
            
            if n==0:
                all_y = np.hstack([smooth_y,current.reshape(-1,1)])
            else:
                _     = np.hstack([smooth_y,current.reshape(-1,1)])
                all_y = np.hstack([all_y,_])

        tobs = []
        for target in y:
            tobs.append( int( min(np.argwhere(np.isnan(target))) ) )
        self.tobs = tobs

        copies      = [y.shape[-1] for y in Y]
        self.copies = copies
            
        #--STORE items
        self.y_means  = y_means
        self.y_scales = y_scales
        self.T        = Y[0].shape[0]  #<--assumption here is Y must be same row for all items in list

        self.all_y  = all_y 
        self.y      = y         #<--Target
        self.Y      = smooth_ys #<--past Y values
        self.X      = X         #<--covariate information 
        
        return y, Y, X, all_y

    
    @staticmethod
    def model( y                 = None
              ,Xhat              = None
              ,R                 = None
              ,B                 = None
              ,dB                = None
              ,B0                = None
              ,Sb                = None
              ,y_past            = None
              ,y_target          = None
              ,target_centers    = None
              ,target_scales     = None
              ,Ls                = None
              ,copies            = None
              ,tobs              = None
              ,target_indicators = None
              ,T                 = None
              ,forecast          = None):


        #--collect helpful parameters
        num_targets       = len(copies)
        L                 = int(sum(Ls))
        
        S_past_total      = int(sum(copies))
        S                 = S_past_total + num_targets

        #--we will flateen target_indicators
        copies_j          = jnp.array(target_indicators)        
        starts            = jnp.concatenate([jnp.array([0]), jnp.cumsum(copies_j + 1)[:-1]])
        target_indicators = starts + copies_j       

        #--We need to build F as a set of P-splines that are rperesented as B @ beta
        T,K = B.shape

        #--This is a smoothness penalty 
        Kp         = 2*jnp.eye(K) - jnp.eye(K,None,-1) - jnp.eye(K,None,1)
        Kp         = Kp.at[0,0].set(1)
        Kp         = Kp.at[-1,-1,].set(1)

        smoothness = numpyro.sample("smoothness", dist.HalfNormal(1./50 ))  
        
        #--draw our betas
        beta       = numpyro.sample("beta", dist.MultivariateNormal(0, precision_matrix = (smoothness**2)*Kp + jnp.eye(K)*10**-6).expand([L]) )
        beta       = beta.T    #<- K X L
        beta       = Sb @ beta #<-- (B @ Sbinv) @ (Sb @ beta). In other words, normalize the columns of our B matrix

        #--Lets assume that the F columns (it the latent time series) are correlated
        Fnorm_Corr = numpyro.sample("Fnorm_Corr", dist.LKJCholesky(L, 2))
        
        F          = B@beta                             #--(T,K) @ (K,L) -> T X L
        sF         = jnp.std(F,0)                       #--We should normalize the columns of F should that X and y can share it
        Fnorm      = F @ jnp.diag(1./sF) @ Fnorm_Corr.T #--TXL
        numpyro.deterministic("Fnorm",Fnorm)
        

        
        #--We will assume that X and y are a sum of the spline trend plus a stochastic component.
        #--This below is the stochastic component
    
        # global_log = numpyro.sample("global_log_rw", dist.Normal(jnp.log(1./2), jnp.log(2)/2))
        # local_log  = numpyro.sample("local_log_rw",  dist.Normal(jnp.log(1.)  , jnp.log(4)/2).expand([S]))
        # sigma_s    = jnp.exp(global_log + local_log)          # (S,)
        
        # rws       = numpyro.sample("rw_innovations"    , dist.Normal(0,1.).expand([T-1,S]))
        # rws       = numpyro.deterministic("rws"        , jnp.cumsum(rws*sigma_s,axis=0))
        # rws       = rws

        global_nu_scale = numpyro.sample("global_nu_scale", dist.Normal(jnp.log(1./2), jnp.log(2)/2))
        nu_scale        = numpyro.sample("nu_scale"       , dist.Normal(jnp.log(1.)   , jnp.log(2)/2).expand([S]))
        z_nus           = numpyro.sample("z_nus"          , dist.Normal(0,1).expand([T,S]))
        nus             = z_nus*jnp.exp( global_nu_scale + nu_scale ) 
        
        global_zeta_scale = numpyro.sample("global_zeta_scale"  , dist.Normal(jnp.log(1./5), jnp.log(2)/2))
        zeta_scale        = numpyro.sample("zeta_scale"       , dist.Normal(jnp.log(1.)  , jnp.log(2)/2).expand([S]))
        z_zetas           = numpyro.sample("z_zetas"          , dist.Normal(0,1).expand([T,S]))
        zetas             = z_zetas*jnp.exp( global_zeta_scale + zeta_scale ) 
        
        l0                = numpyro.sample("l0", dist.Normal(0,1).expand([S])) 
        s0                = jnp.zeros((S,))
        def local_linear_trend(carry, array):
            nu,zeta    = array
            lt_1, st_1 = carry

            st = st_1 + zeta
            lt = lt_1 + st_1 + nu

            return (lt,st) , lt
        _,lt = jax.lax.scan(local_linear_trend, init = (l0,s0) , xs = (nus,zetas) )
        
        #--P matrix so that X = Fnorm @ P
        tau_P     = numpyro.sample("tau_P"   , dist.HalfNormal(1.0))
        lambda_P  = numpyro.sample("lambda_P", dist.HalfCauchy(1.0).expand([L,1]))

        P_blk_raw = numpyro.sample("P_blk_raw", dist.Normal(0,1).expand([L,L]))
        P_blk     = jnp.tril(P_blk_raw, k=-1) + jnp.eye(L)

        P_blk     = P_blk * tau_P * lambda_P  

        P_rest    = numpyro.sample("P_rest", dist.Normal(0,1).expand([L, S_past_total - L]))

        P         = jnp.concatenate([P_rest, P_blk], axis=1)
        P         = jnp.diag(sF) @ P 
        numpyro.deterministic("P", P)
        
        X = Fnorm @ P  #- T X S
        #X = jnp.cumsum( jnp.vstack([X[0,:], jnp.diff(X,axis=0) + rws[:,:-1] ]),  axis=0)
        X = X + lt[:,:-1]

        eps_x = numpyro.sample("eps_x", dist.HalfNormal(3./4) )

        X_target   = jnp.hstack(y_past).reshape(T,S_past_total)
        present    = jnp.isfinite(X_target)

        maskX  = jnp.isfinite(X_target)
        X_fill = jnp.where(maskX, X_target, 0.0)
        with numpyro.handlers.mask(mask=maskX):
            numpyro.sample("X_ll", dist.Normal(X, eps_x), obs=X_fill)

        #--center and scale for y
        centers, scales = [],[]
        for signal, (cents, scals) in enumerate( zip(target_centers, target_scales) ):

            m,s           = jnp.mean(cents), jnp.std(cents)
            center_choice = numpyro.sample(f"center_choice_{signal}", dist.Normal(m,s) )

            m,s           = jnp.mean(jnp.log(1+scals)), jnp.std( jnp.log(1+scals))
            scal_choice   = numpyro.sample(f"scal_choice_{signal}", dist.Normal(m,s) )
            scal_choice   = jnp.exp(scal_choice) - 1

            centers.append(center_choice)
            scales.append(scal_choice)
        
        centers = jnp.hstack(centers)
        scales  = jnp.hstack(scales) 

        Q    = numpyro.sample("Q_raw"    , dist.Normal(0,1).expand([L,1]))
        Q    = numpyro.deterministic("Q" ,jnp.diag(sF) @ Q)

        y  = Fnorm @ Q
        #y  = jnp.cumsum( jnp.vstack([y[0,:], jnp.diff(y,axis=0) + rws[:,-1].reshape(-1,1) ]), axis=0)

        y = y + lt[:,-1].reshape(-1,1)
        numpyro.deterministic("y",y)

        yhat_    = (y)*scales + centers
        numpyro.deterministic("yhat_", yhat_)
        
        eps_y0 = numpyro.sample("eps_y0", dist.HalfNormal(1./2 ))
        eps_y  = eps_y0 * scales

        y_target = jnp.hstack(y_target).reshape(T,1)
        mask     = jnp.isfinite(y_target)
        y_obs    = jnp.where(mask, y_target, 0.0)

        numpyro.deterministic("y_target", y_target)
        
        print(yhat_.shape)
        print(y_obs.shape)
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("y_ll", dist.Normal(yhat_.reshape(T,1), eps_y), obs = y_obs)

        if forecast:
            numpyro.sample("y_pred", dist.Normal(yhat_, eps_y) )

        
    def estimate_factors(self,D):
        u, s, vt            = np.linalg.svd(D, full_matrices=False)
        splain              = np.cumsum(s**2) / np.sum(s**2)
        estimated_factors_D = (np.min(np.argwhere(splain > .95)) + 1)
        return estimated_factors_D, (u,s,vt)

    def fit(self
            , estimate_lmax_x = True
            , estimate_lmax_y = True
            , run_SVI         = True
            , use_anchor      = False):

        y, Y, X     = self.y, self.Y, self.X
        all_y       = self.all_y

        #--SVD for X
        if estimate_lmax_y:
            Ls, us, vts, lambdas = [],[],[], []
            for _ in Y:
                nfactors, (u,s,vt) = self.estimate_factors(_)
                Ls.append(nfactors)
                us.append(u)
                lambdas.append(s)
                vts.append(vt)
                
            self.estimated_factors_y = Ls
            self.us                  = us
            self.lambdas             = lambdas 
            self.vts                 = vts


        from patsy import dmatrix

        def bs_basis_zero_padded(tvals):
            def bs_basis(tvals):
                return np.asarray(
                    dmatrix(
                        "bs(t, knots=knots, degree=3, include_intercept=True, "
                        "lower_bound=lb, upper_bound=ub) - 1",
                        {"t": tvals, "knots": knots, "lb": lb, "ub": ub},
                    )
                )
            
            tvals = np.asarray(tvals, float)
            ok = (tvals >= lb) & (tvals <= ub)

            # build basis for clipped values (any in-range values ok)
            B = np.zeros((len(tvals), bs_basis(np.array([lb])).shape[1]), float)
            if ok.any():
                B_ok = bs_basis(tvals[ok])  # uses fixed knots/bounds
                B[ok] = B_ok
            return B

        # time grid
        T = self.T
        t = jnp.arange(T) #<--move one time unit back

        # choose fixed spline settings
        lb, ub = 0, T-1
        knots = np.linspace(lb, ub, 10)[1:-1]  # example interior knots
        
        B            = bs_basis_zero_padded(t)
        self.B       = B
        self.BtB_inv = jnp.linalg.inv(B.T@B + jnp.eye(B.shape[-1])*10**-6)

        D            = jnp.std(B, 0)
        Sb           = jnp.diag( D  )
        Sbinv        = jnp.diag(1./D)

        B_norm       = B@Sbinv
        dB           = jnp.diff(B_norm, axis=0)
        B0           = B_norm[0,:] 
        B_norm       = B_norm  #<--shift back to time unit zero

        self.B0     = B0
        self.B_norm = B_norm
        self.Sb     = Sb
        self.Sbinv  = Sbinv
        self.dB     = dB
            
        if run_SVI:
            model = self.model

            median_init = init_to_median(num_samples=200)

            us      = self.us
            vts     = self.vts
            lambdas = self.lambdas

            Xhat = jnp.hstack([ u[:,:5] @ ( np.diag(s[:5]) @ vt[:5,:]  )  for u,s,vt in zip(us,lambdas,vts) ])
            R    = jnp.hstack(Y) - Xhat

            self.Xhat = Xhat
            self.R = R
            
            # def _init(site, *, fixed, fallback):
            #     if site["type"] == "sample" and site["name"] in fixed:
            #         return fixed[site["name"]]     # value is on the *constrained* scale
            #     return fallback(site)
            # my_init_loc_fn = partial(_init, fixed=fixed, fallback=median_init)
            
            guide = AutoLowRankMultivariateNormal(  model
                                                  , rank = 50  
                                                    , init_loc_fn=init_to_median(num_samples=200))

            optimizer = Adam(10**-3)  
            svi       = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=3))

            rng_key    = jax.random.PRNGKey(20200320)
            num_steps  = 30*10**3
            svi_result = svi.run( rng_key
                                  ,num_steps
                                  ,y                 = all_y
                                  ,y_past            = Y
                                  ,Xhat              = Xhat
                                  ,B                 = B_norm
                                  ,dB                = dB
                                  ,Sb                = Sb
                                  ,B0                = B0 
                                  ,R                 = R
                                  ,y_target          = y
                                  ,target_centers    = self.y_means
                                  ,target_scales     = self.y_scales
                                  ,Ls                = self.estimated_factors_y
                                  ,copies            = self.copies
                                  ,tobs              = self.tobs
                                  ,T                 = self.T
                                  ,target_indicators = self.target_indicators #<--thius will be flattended inside the model
                                  ,forecast          = None 
            )
            params      = svi_result.params
            self.params = params

            losses      = jnp.asarray(svi_result.losses)
            self.losses = losses

            rng_post          = jax.random.PRNGKey(100915)
            num_draws         = 5*10**3
            posterior_samples = guide.sample_posterior(rng_post,params,sample_shape=(num_draws,))
            self.posterior_samples = posterior_samples

           
        else:
            nuts_kernel = NUTS(self.model, init_strategy = init_to_median(num_samples=100))
            mcmc = MCMC(nuts_kernel
                        , num_warmup  = 5000
                        , num_samples = 8000
                        , num_chains=1
                        , jit_model_args=False)

            mcmc.run(jax.random.PRNGKey(20200320)
                                  ,y                 = all_y
                                  ,y_past            = Y
                                  ,Xhat              = None
                                  ,B                 = self.B_norm
                                  ,dB                = dB
                                  ,R                 = None
                                  ,y_target          = y
                                  ,target_centers    = self.y_means
                                  ,target_scales     = self.y_scales
                                  ,Ls                = self.estimated_factors_y
                                  ,copies            = self.copies
                                  ,tobs              = self.tobs
                                  ,T                 = self.T
                                  ,target_indicators = self.target_indicators #<--thius will be flattended inside the model
                                  ,forecast          = None 
                                  ,extra_fields          =("diverging", "num_steps", "accept_prob", "energy"))

            mcmc.print_summary()
            samples = mcmc.get_samples()
            self.posterior_samples = samples

            #--diagnostics
            extra = mcmc.get_extra_fields()

            print("divergences:", int(extra["diverging"].sum()))

            steps = np.asarray(extra["num_steps"])
            print("num_steps median / 90% / max:", np.median(steps), np.quantile(steps, 0.9), steps.max())

            acc = np.asarray(extra["accept_prob"])
            print("accept_prob mean:", acc.mean(), "min:", acc.min(), "max:", acc.max())

            # Step size (after adaptation)
            print("step_size:", float(mcmc.last_state.adapt_state.step_size))

            if "energy" in extra:
                E = np.asarray(extra["energy"])
                ebfmi = np.mean(np.diff(E)**2) / np.var(E)
                print("E-BFMI:", ebfmi)  # rule of thumb: < 0.3 is concerning

            div_idx = np.where(np.asarray(extra["diverging"]))[0]
            print("divergent draws:", div_idx[:20], "count:", len(div_idx))

            for name in ["A_logit_0", "Q_diags", "global_sigma_rw", "global_f_sigma"]:
                if name in samples:
                    vals = np.asarray(samples[name])
                    print(name, "divergent mean:", vals[div_idx].mean(), "overall mean:", vals.mean())

            #--
            from numpyro.infer import Predictive
            post       = Predictive(  self.model
                                    , { k:v  for k,v in samples.items() if "future" not in k }
                                    , return_sites=["y_pred"])

            pred_samples = post(jax.random.PRNGKey(1)
                                   ,y                 = self.all_y
                                   ,y_past            = self.Y
                                   ,y_target          = self.y
                                   ,target_centers    = self.y_means
                                   ,target_scales     = self.y_scales
                                   ,Ls                = [3]#self.estimated_factors_y
                                   ,copies            = self.copies
                                   ,tobs              = self.tobs
                                   ,T                 = self.T
                                   ,target_indicators = self.target_indicators #<--thius will be flattended inside the model
                                   ,forecast          = True )
            self.pred_samples = pred_samples
       
        return self

    def forecast(self):
        from numpyro.infer import Predictive
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples, return_sites = list(self.posterior_samples.keys()) + ["y_pred", "Y_ll", "y_ll"])#,return_sites = ["y_pred"])

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                                   ,y                 = self.all_y
                                   ,y_past            = self.Y
                                   ,Xhat              = self.Xhat
                                   ,R                 = self.R
                                   ,B                 = self.B_norm
                                   ,dB                = self.dB
                                   ,Sb                = self.Sb
                                   ,B0                = self.B0 
                                   ,y_target          = self.y
                                   ,target_centers    = self.y_means
                                   ,target_scales     = self.y_scales
                                   ,Ls                = self.estimated_factors_y
                                   ,copies            = self.copies
                                   ,tobs              = self.tobs
                                   ,T                 = self.T
                                   ,target_indicators = self.target_indicators #<--thius will be flattended inside the model
                                   ,forecast          = True 
            )
        yhat_draws = pred_samples["y_pred"]      # (draws, T, S)

        self.pred_samples = pred_samples
        self.forecast = yhat_draws
        return yhat_draws
        
        # y, X = self.y, self.X
        # past_y = self.past_y_scaled
        # ncopies = [y.shape[-1] for y in past_y]

        # # Use stored attributes from fit()
        # predictive = Predictive(puca.model, posterior_samples=self.post_samples)
        # pred_samples = predictive(jax.random.PRNGKey(20200320)
        #                           , Y=y
        #                           , X=X
        #                           , ncopies=ncopies
        #                           , LMAX_x=self.estimated_factors_x
        #                           , forecast=True
        #                           , centers=self.centers_flat
        #                           , scales=self.scales_flat
        #                           , us            = self.u_x[:,:estimated_factors_x] 
        #                           , anchor=self.anchor)


        # pred_samples = predictive(jax.random.PRNGKey(20200320)
        #                           , Y=y
        #                           , X=X
        #                           , ncopies=ncopies
        #                           , LMAX_x=estimated_factors_x
        #                           , forecast=True
        #                           , centers=centers
        #                           , scales=scales
        #                           , observations=[np.min(np.argwhere(np.isnan(col))) for col in y.T]
        #                           , test=[np.min(np.argwhere(np.isnan(col)))+1 for col in y.T]
        #                           , anchor= anchor)


        
        
        self.forecasts = pred_samples
        return pred_samples

if __name__ == "__main__":

    pass
