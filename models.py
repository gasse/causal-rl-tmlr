import typing
import torch
import numpy as np
from utils import Dataset, print_log, collate_fn_pad_episodes, collate_fn_pad_episodes_weights


class AugmentedModel(torch.nn.Module):

    """ Augmented Model Base Class """

    def __init__(self, with_done):
        super().__init__()
        self.with_done = with_done

    def log_q_s(self, s=None):
        raise NotImplementedError

    def log_q_snext_sa(self, a, s=None, snext=None):
        raise NotImplementedError

    def log_q_o_s(self, o, s=None):
        raise NotImplementedError

    def log_q_r_s(self, r, s=None):
        raise NotImplementedError

    def log_q_d_s(self, d, s=None):
        raise NotImplementedError

    def log_q_a_s(self, a, s=None):
        raise NotImplementedError

    def log_q_s_h(self, regime, loq_q_sprev_hprevi, a, o, r, d):

        assert (loq_q_sprev_hprevi is None) == (a is None)

        # hprev = (o_0, r_0, d_0, a_0, ..., o_t-1, r_t-1, d_t-1)
        # sprev = s_t-1
        # a = a_t-1
        # o = o_t
        # r = r_t
        # d = d_t
        # h =     (o_0, r_0, d_0, a_0, ..., o_t-1, r_t-1, d_t-1, a_t-1, o_t, r_t, d_t)
        # s = s_t
        # i = regime

        no_hprev = (a is None)

        if no_hprev:
            # (batch_size, s_nvals)
            log_q_s_hprevai = self.log_q_s()

        else:

            # (batch_size, sprev_nvals)
            log_q_a_sprevi = self.log_q_a_s(a=a)
            log_q_a_sprevi = torch.where((d==1).unsqueeze(-1), 0, log_q_a_sprevi)  # discard actions if done=True
            log_q_a_sprevi = torch.where((regime==1).unsqueeze(-1), 0, log_q_a_sprevi)  # discard actions in interventional regime

            # (batch_size, sprev_nvals)
            log_q_spreva_hprevi = loq_q_sprev_hprevi + log_q_a_sprevi

            # (batch_size,)
            log_q_a_hprevi = torch.logsumexp(log_q_spreva_hprevi, dim=-1)

            # (batch_size, sprev_nvals)
            log_q_sprev_hprevai = log_q_spreva_hprevi - log_q_a_hprevi.unsqueeze(-1)

            # (batch_size, sprev_nvals, s_nvals)
            log_q_s_sprevai = self.log_q_snext_sa(a=a)

            # (batch_size, sprev_nvals, s_nvals)
            loq_q_sprevs_hprevai = log_q_sprev_hprevai.unsqueeze(-1) + log_q_s_sprevai

            # (batch_size, s_nvals)
            log_q_s_hprevai = torch.logsumexp(loq_q_sprevs_hprevai, dim=-2)

        log_q_o_s = self.log_q_o_s(o=o)
        log_q_r_s = self.log_q_r_s(r=r)
        log_q_d_s = self.log_q_d_s(d=d) if self.with_done else 0

        # (batch_size, s_nvals)
        log_q_ord_s = log_q_o_s + log_q_r_s + log_q_d_s

        # (batch_size, s_nvals)
        log_q_sord_hprevai = log_q_s_hprevai + log_q_ord_s

        # (batch_size,)
        log_q_ord_hprevai = torch.logsumexp(log_q_sord_hprevai, dim=-1)

        # (batch_size, s_nvals)
        log_q_s_hi = log_q_sord_hprevai - log_q_ord_hprevai.unsqueeze(-1)

        return log_q_s_hi

    @torch.jit.export
    def log_prob_joint(self, regime: torch.Tensor, episode: typing.List[torch.Tensor], states: torch.Tensor):
        log_prob = 0

        n_transitions = len(episode) // 4
        for t in range(n_transitions + 1):

            # s_t, o_t, r_t, d_t
            state, obs, reward, done = states[t], episode[4*t], episode[4*t+1], episode[4*t+2]

            if t == 0:
                was_done = torch.zeros_like(done)

                # (batch_size, )
                log_q_s_saiprev = self.log_q_s(s=state)

            else:

                # safety fix, in case a done flag goes back down
                done = torch.max(was_done, done)

                # s_t-1, a_t-1
                state_prev, action_prev = states[t-1], episode[4*t-1]

                # (batch_size, )
                log_q_s_saiprev = self.log_q_snext_sa(a=action_prev, s=state_prev, snext=state)

            # (batch_size, )
            log_q_o_s = self.log_q_o_s(o=obs, s=state)
            log_q_r_s = self.log_q_r_s(r=reward, s=state)

            # (batch_size, )
            log_q_d_s = self.log_q_d_s(d=done, s=state) if self.with_done else 0

            # a_t (if any)
            if t < n_transitions:
                action = episode[4*(t+1)-1]

                # (batch_size, )
                log_q_a_si = self.log_q_a_s(a=action, s=state)
                log_q_a_si = torch.where((done==1).unsqueeze(-1), 0, log_q_a_si)  # discard actions if done=True
                log_q_a_si = torch.where((regime==1).unsqueeze(-1), 0, log_q_a_si)  # discard actions in interventional regime
            else:
                log_q_a_si = 0

            # (batch_size, )
            log_q_sorda_saiprev = log_q_s_saiprev + log_q_o_s + log_q_r_s + log_q_d_s + log_q_a_si

            # discard transitions after done=True (might happen due to padding)
            log_q_sorda_saiprev = torch.where(was_done==1, 0, log_q_sorda_saiprev)

            # (batch_size, )
            log_prob = log_prob + log_q_sorda_saiprev

            if t == 0:
                # (batch_size, )
                log_prob = log_q_sorda_saiprev
            else:
                # (batch_size, )
                log_prob = torch.where(log_prob.isinf(), log_prob, log_prob + log_q_sorda_saiprev)  # bugfix, otherwise NaNs will appear

            was_done = done

        return log_prob

    @torch.jit.export
    def log_prob(self, regime: torch.Tensor, episode: typing.List[torch.Tensor], return_loq_q_s_hai: bool=False):

        # if requested, store all q(s_t|h_t,a_t,regime) and q(s_t+1|h_t,a_t,regime) during forward
        if return_loq_q_s_hai:
            seq_loq_q_s_hai = []

        log_prob = 0

        n_transitions = len(episode) // 4
        assert len(episode) == 4 * n_transitions + 3

        seq_o, seq_r, seq_d, seq_a = [], [], [], []

        for t in range(n_transitions + 1):

            # a_t if t!=T
            if t < n_transitions:
                # o_t, r_t, d_t, a_t
                o, r, d, a = episode[4*t:4*t+4]
            else:
                # o_T, r_T, d_T
                o, r, d = episode[4*t:4*t+3]
                d = torch.ones_like(d)  # safety fix, done flag should be raised at the end
                # a_T
                a = torch.zeros_like(o)  # fake final action, for convenience

            if t > 0:
                d = torch.maximum(d, seq_d[-1])  # safety fix, done flag should be persistent

            seq_o.append(o)
            seq_r.append(r)
            seq_d.append(d)
            seq_a.append(a)

        # (n_trans+1, batch_size)
        seq_o = torch.stack(seq_o, dim=0)
        seq_r = torch.stack(seq_r, dim=0)
        seq_d = torch.stack(seq_d, dim=0)
        seq_a = torch.stack(seq_a, dim=0)
        seq_was_d = torch.cat([torch.zeros_like(seq_d[-1:]), seq_d[:-1]], dim=0)

        ## Pre-compute log-probs

        # (n_trans+1, batch_size, s_nvals)
        log_q_o_s = self.log_q_o_s(o=seq_o.reshape(-1))
        log_q_o_s = log_q_o_s.view((*seq_o.size(), log_q_o_s.size(1)))

        # (n_trans+1, batch_size, s_nvals)
        log_q_r_s = self.log_q_r_s(r=seq_r.reshape(-1))
        log_q_r_s = log_q_r_s.view((*seq_r.size(), log_q_r_s.size(1)))

        # (n_trans+1, batch_size, s_nvals)
        if self.with_done:
            log_q_d_s = self.log_q_d_s(d=seq_d.reshape(-1))
            log_q_d_s = log_q_d_s.view((*seq_d.size(), log_q_d_s.size(1)))
        else:
            log_q_d_s = torch.zeros_like(log_q_r_s)

        # (n_trans+1, batch_size, s_nvals)
        log_q_ord_s = log_q_o_s + log_q_r_s + log_q_d_s

        # ignore transitions after done flag raised
        log_q_ord_s = torch.where((seq_was_d==1).unsqueeze(-1), 0, log_q_ord_s)

        # (n_trans+1, batch_size, s_nvals)
        log_q_a_si = self.log_q_a_s(a=seq_a.reshape(-1))
        log_q_a_si = log_q_a_si.view((*seq_a.size(), -1))

        # ignore actions if done flag raised
        log_q_a_si = torch.where((seq_d==1).unsqueeze(-1), 0, log_q_a_si)

        # states do not affect actions in interventional regime
        log_q_a_si = torch.where((regime==1).view(1, -1, 1), 0, log_q_a_si)

        # (n_trans+1, batch_size, s_nvals)
        log_q_orda_si = log_q_ord_s + log_q_a_si

        # (n_trans+1, batch_size, s_nvals, snext_nvals)
        loq_q_snext_sa = self.log_q_snext_sa(a=seq_a.reshape(-1))
        loq_q_snext_sa = loq_q_snext_sa.view((*seq_a.size(), loq_q_snext_sa.size(1), loq_q_snext_sa.size(2)))

        # (1, s_nvals)
        log_q_snext_hai = self.log_q_s().unsqueeze(0)

        for t in range(n_transitions + 1):

            # haiprev = (regime, o_0, r_0, d_0, a_0, ..., o_t-1, r_t-1, d_t-1, a_t-1)

            log_q_s_haiprev = log_q_snext_hai

            # (batch_size, s_nvals)
            log_q_sorda_haiprev = log_q_s_haiprev + log_q_orda_si[t]

            # (batch_size, )
            log_q_orda_haiprev = torch.logsumexp(log_q_sorda_haiprev, dim=-1)

            if t == 0:
                # (batch_size, )
                log_prob += log_q_orda_haiprev
            else:
                # (batch_size, )
                log_prob = torch.where(log_prob.isinf(), log_prob, log_prob + log_q_orda_haiprev)  # prevent NaNs

            # hai = (regime, o_0, r_0, d_0, a_0, ..., o_t, r_t, d_t, a_t)

            # (batch_size, s_nvals)
            log_q_s_hai = log_q_sorda_haiprev - log_q_orda_haiprev.unsqueeze(1)

            # snext = s_t+1

            if t < n_transitions:
                # (batch_size, s_nvals, snext_nval)
                log_q_ssnext_hai = log_q_s_hai.unsqueeze(2) + loq_q_snext_sa[t]
                # (batch_size, snext_nvals)
                log_q_snext_hai = torch.logsumexp(log_q_ssnext_hai, dim=1)

            else:
                log_q_ssnext_hai = None
                log_q_snext_hai = None

            # if requested, store all q(s_t|h_t,a_t,regime) and q(s_t+1|h_t,a_t,regime) during forward
            if return_loq_q_s_hai:
                seq_loq_q_s_hai.append((log_q_s_hai, log_q_ssnext_hai, log_q_snext_hai))

        if return_loq_q_s_hai:
            return log_prob, seq_loq_q_s_hai
        else:
            return log_prob

    @torch.jit.export
    def sample_states(self, regime: torch.Tensor, episode: typing.List[torch.Tensor]):

        with torch.no_grad():

            # collect all q(s_t | h_t,a_t,regime) with a forward pass
            _, seq_log_q_s_hai = self.log_prob(regime, episode, return_loq_q_s_hai=True)

            # collect all s_t ~ q(s_t | h_t,a_t,regime,s_t+1) with a backward pass
            states = []
            n_transitions = len(episode) // 4
            for t in range(n_transitions, -1, -1):
                log_q_s_hai, log_q_ssnext_hai, log_q_snext_hai = seq_log_q_s_hai[t]

                if t == n_transitions:
                    # (batch_size, snext_nvals)
                    log_q_s_haisnext = log_q_s_hai

                else:

                    snext = s

                    # (batch_size, snext_nvals, s_nvals)
                    log_q_s_haisnext = (log_q_ssnext_hai - log_q_snext_hai.unsqueeze(1)).permute(0, 2, 1)

                    # (batch_size, s_nvals)
                    log_q_s_haisnext = log_q_s_haisnext.gather(dim=1, index=snext.view(snext.size(0), 1, 1).expand(log_q_s_haisnext.size(0), 1, log_q_s_haisnext.size(2))).squeeze(1)

                s = torch.distributions.categorical.Categorical(logits=log_q_s_haisnext).sample()

                states.insert(0, s)

        return states

    @torch.jit.export
    def rsample_states(self, regime: torch.Tensor, episode: typing.List[torch.Tensor]):
        raise NotImplementedError

    @torch.jit.export
    def loss_nll(self, regime: torch.Tensor, episode: typing.List[torch.Tensor]):
        return -self.log_prob(regime, episode)

    @torch.jit.export
    def loss_em(self, regime: torch.Tensor, episode: typing.List[torch.Tensor]):
        states = self.sample_states(regime, episode)
        return -self.log_prob_joint(regime, episode, states)


class TabularAugmentedModel(AugmentedModel):
    
    """ Learnable Augmented Model using tabular probability distribution parameters. """

    def __init__(self, s_nvals, o_nvals, a_nvals, r_nvals, with_done):
        super().__init__(with_done=with_done)
        self.s_nvals = s_nvals
        self.o_nvals = o_nvals
        self.a_nvals = a_nvals
        self.r_nvals = r_nvals

        # p(s_0)
        self.params_s = torch.nn.Parameter(torch.empty([s_nvals]))
        torch.nn.init.normal_(self.params_s)

        # p(s_t+1 | s_t, a_t)
        self.params_s_sa = torch.nn.Parameter(torch.empty([s_nvals, a_nvals, s_nvals]))
        torch.nn.init.normal_(self.params_s_sa)

        # p(o_t | s_t)
        self.params_o_s = torch.nn.Parameter(torch.empty([s_nvals, o_nvals]))
        torch.nn.init.normal_(self.params_o_s)

        # p(r_t | s_t)
        self.params_r_s = torch.nn.Parameter(torch.empty([s_nvals, r_nvals]))
        torch.nn.init.normal_(self.params_r_s)

        # p(d_t | s_t)
        self.params_d_s = torch.nn.Parameter(torch.empty([s_nvals]))
        torch.nn.init.normal_(self.params_d_s)

        # p(a_t | s_t, i=0)
        self.params_a_s = torch.nn.Parameter(torch.empty([s_nvals, a_nvals]))
        torch.nn.init.normal_(self.params_a_s)

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_s(self, s: typing.Optional[torch.Tensor]=None):

        """ Log proba of state distribution p(s) """ 

        log_q_s = torch.nn.functional.log_softmax(self.params_s, dim=-1)

        if s is not None:
            assert s.ndim == 1  # batched
            log_q_s = log_q_s[s]

        return log_q_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_snext_sa(self, a: torch.Tensor,
                       s: typing.Optional[torch.Tensor]=None,
                       snext: typing.Optional[torch.Tensor]=None):

        """ Log proba of state transition distribution p(s|s, a). """ 

        # (s_nvals, a_nvals, s_nvals)
        log_q_snext_sa = torch.nn.functional.log_softmax(self.params_s_sa, dim=-1)
        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert a.ndim == 1  # batched
        indices.insert(0, a)
        log_q_snext_sa = log_q_snext_sa.permute(1, 0, 2)

        if snext is not None:
            assert snext.ndim == 1  # batched
            indices.insert(0, snext)
            log_q_snext_sa = log_q_snext_sa.permute(2, 0, 1)

        log_q_snext_sa = log_q_snext_sa[indices]

        return log_q_snext_sa

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_o_s(self, o: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional observation distribution from state p(o|s). """ 

        log_q_o_s = torch.nn.functional.log_softmax(self.params_o_s, dim=-1)

        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert o.ndim == 1  # batched
        indices.insert(0, o)
        log_q_o_s = log_q_o_s.permute(1, 0)

        log_q_o_s = log_q_o_s[indices]

        return log_q_o_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_a_s(self, a: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional action distribution from state p(a|s). """ 

        log_q_a_s = torch.nn.functional.log_softmax(self.params_a_s, dim=-1)

        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert a.ndim == 1  # batched
        indices.insert(0, a)
        log_q_a_s = log_q_a_s.permute(1, 0)

        log_q_a_s = log_q_a_s[indices]

        return log_q_a_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_r_s(self, r: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional reward distribution from state p(r|s). """ 

        log_q_r_s = torch.nn.functional.log_softmax(self.params_r_s, dim=-1)

        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert r.ndim == 1  # batched
        indices.insert(0, r)
        log_q_r_s = log_q_r_s.permute(1, 0)

        log_q_r_s = log_q_r_s[indices]

        return log_q_r_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_d_s(self, d: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional flagDone distribution from state p(d|s). """

        assert d.ndim == 1  # batched

        if s is not None:
            assert s.ndim == 1  # batched
            params_d_s = self.params_d_s[s]

        else:
            d, params_d_s = torch.broadcast_tensors(
                d.unsqueeze(1),
                self.params_d_s.unsqueeze(0))

        log_q_d_s = -torch.nn.functional.binary_cross_entropy_with_logits(params_d_s, d.to(dtype=params_d_s.dtype), reduction='none')

        return log_q_d_s

    def get_probs(self):

        settings = {
            "p_s": torch.nn.functional.softmax(self.params_s.detach(), dim=-1),
            "p_o_s": torch.nn.functional.softmax(self.params_o_s.detach(), dim=-1),
            "p_r_s": torch.nn.functional.softmax(self.params_r_s.detach(), dim=-1),
            "p_d_s": torch.sigmoid(self.params_d_s.detach()) if self.with_done else torch.zeros_like(self.params_d_s),
            "p_snext_sa": torch.nn.functional.softmax(self.params_s_sa.detach(), dim=-1),
        }

        return settings


def fit_model(m, train_data, valid_data, loss_type='nll',
              n_epochs=200, epoch_size=100, batch_size=16,
              lr=1e-2, patience=10, log=False, logfile=None, min_int_ratio=0.0, threshold=1e-4):

    # infer the device from the model
    device = next(m.parameters()).device

    if log:
        print_log(f"Model loss_type: {loss_type}", logfile)
        print_log(f"Model n_epochs: {n_epochs}", logfile)
        print_log(f"Model epoch_size: {epoch_size}", logfile)
        print_log(f"Model batch_size: {batch_size}", logfile)
        print_log(f"Model lr: {lr}", logfile)
        print_log(f"Model patience: {patience}", logfile)
        print_log(f"Model device: {device}", logfile)
        print_log(f"Model min_int_ratio: {min_int_ratio}", logfile)

    def compute_weights(data):
        nint = np.sum([regime == 1 for regime, _ in data])
        nobs = len(data) - nint
        int_ratio = nint / (nint + nobs)

        if int_ratio >= min_int_ratio:
            weights = [1] * len(data)
        else:
            weights = [(1 - min_int_ratio) / nobs, min_int_ratio / nint]  # obs, int
            weights = [weights[int(regime)] for regime, _ in data]

        return weights

    train_weights = compute_weights(train_data)
    valid_weights = compute_weights(valid_data)

    # Build training and validation datasets and dataloaders
    train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, replacement=True, num_samples=epoch_size*batch_size)
    train_dataset = Dataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn_pad_episodes)

    valid_dataset = Dataset(list(zip(valid_data, valid_weights)))  # to reweight the loss
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn_pad_episodes_weights)

    # Adam Optimizer with learning rate lr
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    # Scheduler. Reduce learning rate on plateau.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=log, threshold=threshold)

    # Early stopping
    best_valid_loss = float("Inf")
    best_parameters = m.state_dict().copy()
    best_epoch = -1

    # Start training loop
    for epoch in range(n_epochs + 1):

        # Set initial training loss as +inf 
        if epoch == 0:
            train_loss = float("Inf")

        else:
            train_loss = 0
            train_nsamples = 0

            for batch in train_loader:
                regime, episode = batch
                regime = regime.to(device)
                episode = [tensor.to(device) for tensor in episode] 

                batch_size = regime.shape[0]

                if loss_type == 'em':
                    loss = m.loss_em(regime, episode).mean()
                elif loss_type == 'nll':
                    loss = m.loss_nll(regime, episode).mean()
                elif loss_type == 'elbo':
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_size
                train_nsamples += batch_size

            train_loss /= train_nsamples

        # validation
        valid_loss = 0
        valid_nsamples = 0

        for batch in valid_loader:
            (regime, episode), weight = batch
            regime = regime.to(device)
            episode = [tensor.to(device) for tensor in episode] 
            weight = weight.to(device)

            batch_size = regime.shape[0]

            with torch.no_grad():
                
                loss = m.loss_nll(regime, episode)
                loss = (loss * weight).sum()  # re-weighting the loss here

            valid_loss += loss.item()
            valid_nsamples += weight.sum().item()

        valid_loss /= valid_nsamples

        if log:
            print_log(f"Model epoch {epoch:04d} / {n_epochs:04d} train_loss={train_loss:0.4f} valid_loss={valid_loss:0.4f}", logfile)

        # check for best model
        if valid_loss < (best_valid_loss - threshold):
            best_valid_loss = valid_loss
            best_parameters = m.state_dict().copy()
            best_epoch = epoch

        # check for early stopping
        if epoch > best_epoch + 2 * patience:
            if log:
                print_log(f"Model {epoch-best_epoch} epochs without improvement, stopping.", logfile)
            break

        scheduler.step(valid_loss)

    # restore best model
    m.load_state_dict(best_parameters)

def evaluate_model(m, test_data, batch_size=16):

    # infer the device from the model
    device = next(m.parameters()).device

    test_dataset = Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_pad_episodes)

    test_loss = 0.
    test_nsamples = 0

    for batch in test_loader:
        regime, episode = batch
        regime = regime.to(device)
        episode = [tensor.to(device) for tensor in episode] 

        with torch.no_grad():
            test_loss += m.loss_nll(regime, episode).sum().item()

        test_nsamples += regime.shape[0]

    test_loss /= test_nsamples

    return test_loss
