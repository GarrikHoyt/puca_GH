#mcandrew

# Optimization 1: Move all imports to module level
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

            past_smooth       =  self.smooth_gaussian_anchored(past,1)
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

        def ordered_softmax(S_spec, L_spec, base_sd=0.5):
            # first logit free
            a0 = numpyro.sample("A_logit0", dist.Normal(0., 1.).expand([S_spec, 1]))
            # positive increments ensure logits strictly decrease
            deltas = numpyro.sample("A_deltas", dist.HalfNormal(base_sd).expand([S_spec, L_spec - 1]))
            logits_tail = a0 - jnp.cumsum(deltas, axis=-1)   # a0 - d1, a0 - (d1+d2), ...
            logits = jnp.concatenate([a0, logits_tail], axis=-1)  # (S_spec, L_spec) decreasing
            return jax.nn.softmax(logits, axis=-1)
        
        num_targets       = len(copies)
        L                 = int(sum(Ls))
        S_past_total      = int(sum(copies))
        S                 = S_past_total + num_targets

        #--we will flateen target_indicators
        copies_j          = jnp.array(target_indicators)        
        starts            = jnp.concatenate([jnp.array([0]), jnp.cumsum(copies_j + 1)[:-1]])
        target_indicators = starts + copies_j       

        #--Construct the factors for all time series data. They can be built together for all signals
        conc = numpyro.sample("conc",dist.Exponential(1./2))
        CORR = numpyro.sample("CORR", dist.LKJCholesky(L,1+conc))

        global_f_sigma = numpyro.sample("global_f_sigma"      , dist.Normal( jnp.log(1.), jnp.log(2)/2 ))
        f_sigma        = numpyro.sample("f_sigma"             , dist.Normal( jnp.log(1.), jnp.log(2)/2).expand([L]) )

        f_scale       = numpyro.deterministic("f_scale"       , jnp.exp(global_f_sigma+f_sigma))

        f_scale_start = numpyro.sample("f_scale_start"        , dist.Normal( 0,1 ))
        f_scale_start = jnp.exp(f_scale_start)
        
        f_innov_start = numpyro.sample("z_f_innov_start", dist.Normal(0,1).expand([1  ,L])) 
        f_innov_start = (f_innov_start @ CORR.T)*(f_scale_start)
        
        f_innovations  = numpyro.sample("f_innovations_raw", dist.Normal(0,1).expand([T-1,L])) #< 1 X L
        f_innovations  = (f_innovations @ CORR.T) * f_scale
        f_innovations  = numpyro.deterministic("f_innovations", f_innovations)

        #--Need to build Q as the factor transision matrix (think of a VAR-like construciton here)
        #--Q is a LXL transition matrix. Implicitly, we assume all the factors are linked and so the signals ae correlated.
        Q_diags      = numpyro.sample("Q_diags", dist.Beta(2,2).expand([L]))
        Q            = jnp.eye(L) * Q_diags*(1-0.02) #<--make sure its not too close to 1
 
        #--Need to build A matrix that computes as weighted combination of the factors. IE the A matrix maps from factors to observation signals 
        A_full            = jnp.zeros((S_past_total + len(copies), L)) #<--need to make extra seasons to account for the targets
        logits_sd         = 1./2

        l0, s0 = 0, 0
        for signal, ( L_spec, S_spec) in enumerate(zip(Ls, copies)):
            S_spec+=1 #<--this is how we add-in the new, incomplete season
            if L_spec == 1:
                A_raw = jnp.ones((S_spec, 1))
            else:
                with numpyro.handlers.scope(prefix=f"A_{signal}"):
                    A_raw = ordered_softmax(S_spec, L_spec, base_sd=0.3)
            A_full = A_full.at[ s0:s0+S_spec, l0:l0+L_spec ].set(A_raw)
            
            l0 += L_spec
            s0 += S_spec
            
        A = numpyro.deterministic("A", A_full.T) #<-- this is dim S X L (ie it inclues the target signal seasons
        
        #--The past data is z-scored but the target data is in natural scale.
        #--We assume here that the scale of the targets are related to the past scales
        all_scales = jnp.ones(S)
        for signal,(scales,target_indicator) in enumerate(zip(target_scales,target_indicators)):
            scale_choice = numpyro.sample(f"scale_{signal}", dist.StudentT(5, jnp.mean(jnp.log(scales)), jnp.std(jnp.log(scales)) ) )
            all_scales   = all_scales.at[target_indicator].set( jnp.exp(scale_choice))
        numpyro.deterministic("all_scales",all_scales)
            
        all_centers = jnp.zeros(S)
        for signal,(centers,target_indicator) in enumerate(zip(target_centers,target_indicators)):
            center_choice = numpyro.sample(f"center_{signal}", dist.StudentT(5, jnp.mean( centers), jnp.std(centers) ) )
            all_centers   = all_centers.at[target_indicator].set( center_choice)
        numpyro.deterministic("all_centers",all_centers)


        #--Level
        global_sigma_level = numpyro.sample("global_sigma_level", dist.Normal( jnp.log(1./3), jnp.log(2)/2 ))
        local_sigma_level  = numpyro.sample("local_sigma_level" , dist.Normal( 0. , jnp.log(2.)/2).expand([S]) )
        local_sigma_level  = local_sigma_level - jnp.mean(local_sigma_level)
        sigma_level        = numpyro.deterministic( "sigma_level", jnp.exp(global_sigma_level + local_sigma_level))

        Z_level = numpyro.sample("Z_level", dist.Normal(0,1).expand([T-1,S]) )
        level_innovations = Z_level * sigma_level
        #level_start =  numpyro.sample("level_start", dist.Normal(0, 1).expand([S,])) 
        level_start = jnp.zeros((S,))
        
        #--Slope 
        global_sigma_slope = numpyro.sample("global_sigma_slope", dist.Normal( jnp.log(1./3), jnp.log(2)/2 ))
        #global_sigma_slope = numpyro.sample("global_sigma_slope", dist.Exponential( 1./(1./2) ))
        
        local_sigma_slope  = numpyro.sample("local_sigma_slope" , dist.Normal( 0. , jnp.log(2.)/2).expand([S]) )
        local_sigma_slope  = local_sigma_slope - jnp.mean(local_sigma_slope)
        
        sigma_slope        = numpyro.deterministic( "sigma_slope", jnp.exp(global_sigma_slope + local_sigma_slope))
        #sigma_slope        = numpyro.deterministic( "sigma_slope", global_sigma_slope* jnp.exp(local_sigma_slope))

        
        Z_slope = numpyro.sample("Z_slope", dist.Normal(0,1).expand([T-1,S]) )
        slope_innovations = Z_slope * sigma_slope
        #slope_start        = numpyro.sample("slope_start", dist.Normal(0, jnp.log(2)/2 ).expand([S,])) #1,S
        slope_start = jnp.zeros((S,))
        
       
        def evolve_latent_state(carry, array, Q=None):
            f_past,level_past, slope_past  = carry
            f_err ,level_err, slope_err  = array

            slope_next = slope_past              + slope_err
            level_next = level_past + slope_past + level_err
            f_next     =  Q@f_past  + f_err

            y_next =  f_next @ A  + level_next
            
            return (f_next,level_next,slope_next), y_next

        _, yhat = jax.lax.scan( lambda x,y: evolve_latent_state(x,y,Q)
                                , init = ( f_innov_start.reshape(-1,), level_start, slope_start )
                                , xs   = (f_innovations , level_innovations, slope_innovations)  )
        
        yhat = jnp.vstack([ f_innov_start@A + level_start, yhat])
        
        global_sigma_y = numpyro.sample("global_sigma_y"  , dist.Normal(jnp.log(1./2), jnp.log(2)/2 ).expand([ num_targets,1])  )

        sigma_y = jnp.zeros(S)
        holder = 0
        for s,(gs,c) in enumerate(zip(global_sigma_y,copies)):
            sy        = numpyro.sample(f"sigma_y_{s}" , dist.Normal(0,jnp.log(2)/2).expand([c+1]) ) #<--add one to account for the target
            sigma_y = sigma_y.at[holder:holder+c+1].set(sy+gs)
            holder+=c+1
        sigma_y = jnp.exp(sigma_y)
        
        mu = numpyro.deterministic("mu", (yhat)*all_scales + all_centers)
        s  = numpyro.deterministic("s" ,  all_scales*sigma_y)

        present = jnp.isfinite(y)
        y_filled = jnp.where(present, y, 0.0)

        with numpyro.handlers.mask(mask=present):
            numpyro.sample("ll", dist.Normal( mu, s), obs = y_filled )

        if forecast is not None:
            mu_target  = mu[:, jnp.array(target_indicators)]
            s          = s[jnp.array(target_indicators)]
            yhat_preds_raw = numpyro.sample("y_pred", dist.Normal(mu_target ,s))


    def estimate_factors(D):
        try:
            u, s, vt            = np.linalg.svd(D, full_matrices=False)
            splain              = np.cumsum(s**2) / np.sum(s**2)
            estimated_factors_D = (np.min(np.argwhere(splain > .95)) + 1)
        except:
            estimated_factors_D = 1
        return estimated_factors_D

    def fit(self
            , estimate_lmax_x = True
            , estimate_lmax_y = True
            , run_SVI         = True
            , use_anchor      = False):

        y, Y, X     = self.y, self.Y, self.X
        all_y       = self.all_y

        #--SVD for X
        if estimate_lmax_y:
            Ls = []
            for _ in Y:
                try:
                    Ls.append(self.estimate_factors(_))
                except:
                    Ls.append( 1 )
            self.estimated_factors_y = Ls

        if run_SVI:
            model = self.model
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
            full_vars = [ ("f_innovations_raw",), ("Z",)]

            nuts_kernel = NUTS(model, dense_mass=full_vars, init_strategy = init_to_median(num_samples=100))
            mcmc = MCMC(nuts_kernel
                        , num_warmup  = 5000
                        , num_samples = 8000
                        , num_chains=1
                        , jit_model_args=False)

            mcmc.run(jax.random.PRNGKey(20200320)
                     ,y                     = y 
                     ,y_past                = y_past
                     ,y_target              = y_target
                     ,target_centers        = y_means
                     ,target_scales         = y_scales
                     ,Ls                    = Ls
                     ,copies                = copies
                     ,tobs                  = tobs
                     ,T                     = y.shape[0] 
                     ,forecast              = None
                     ,extra_fields=("diverging", "num_steps", "accept_prob", "energy"))

            mcmc.print_summary()
            samples = mcmc.get_samples()

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
            post       = Predictive(  model
                                    , { k:v  for k,v in samples.items() if "future" not in k }
                                    , return_sites=["y_pred"])

            post_samples = post(jax.random.PRNGKey(1)
                                , y = y 
                                ,y_past                = y_past
                                ,y_target              = y_target
                                ,target_centers        = y_means
                                ,target_scales         = y_scales
                                ,Ls                    = Ls
                                ,copies                = copies
                                ,tobs                  = tobs
                                ,T                     = y.shape[0] 
                                ,forecast              = True)
       
        return self

    def forecast(self):
        from numpyro.infer import Predictive
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples,return_sites = ["y_pred"])

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                                   ,y                 = self.all_y
                                   ,y_past            = self.Y
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
