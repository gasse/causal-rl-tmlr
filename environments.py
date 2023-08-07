import torch


class TabularPOMDP(torch.nn.Module):

    def __init__(self, p_s, p_o_s, p_r_s, p_s_sa, p_d_s):
        """ Tabular POMDP environment.
            p_s: p(s) initial distribution of the hidden state.
            p_o_s: p(o|s) distribution of observation given hidden state.
            p_r_s: p(r|s) distribution of reward given hidden state.
            p_r_s: p(r|s) distribution of done flag given hidden state.
            p_s_sa: p(s|s,a) transition distribution of next hidden state given current hidden state and action.
            max_episode_length: maximum length of an episode.
        """
        super().__init__()

        # POMDP dynamics (registered as buffers so that they can be moved to GPU)
        self.register_buffer('p_s', torch.as_tensor(p_s))  # size (s_nvals,)
        self.register_buffer('p_s_sa', torch.as_tensor(p_s_sa))  # size (s_nvals, a_nvals, s_nvals)
        self.register_buffer('p_o_s', torch.as_tensor(p_o_s))  # size (s_nvals, o_nvals)
        self.register_buffer('p_r_s', torch.as_tensor(p_r_s))  # size (s_nvals, r_nvals)
        self.register_buffer('p_d_s', torch.as_tensor(p_d_s))  # size (s_nvals,)

        # sanity checks
        assert self.p_s.ndim == 1
        assert self.p_s_sa.ndim == 3
        assert self.p_o_s.ndim == 2
        assert self.p_r_s.ndim == 2
        assert self.p_s.shape[0] == self.p_s_sa.shape[0]
        assert self.p_s.shape[0] == self.p_s_sa.shape[2]
        assert self.p_s.shape[0] == self.p_o_s.shape[0]
        assert self.p_s.shape[0] == self.p_r_s.shape[0]
        assert self.p_s.shape[0] == self.p_d_s.shape[0]

        self.s_nvals = self.p_s.shape[0]
        self.a_nvals = self.p_s_sa.shape[1]
        self.o_nvals = self.p_o_s.shape[1]
        self.r_nvals = self.p_r_s.shape[1]

        # Initialize current time step
        self.current_step = -1

    def reset(self, batch_size=None):

        if batch_size is None:
            self.batch_shape = torch.Size()
        else:
            self.batch_shape = torch.Size([batch_size])

        self.current_step = 0

        return self._sample_next_ordi()

    def step(self, action):

        self.a = torch.as_tensor(action)

        assert self.a.size() == self.batch_shape

        self.current_step += 1

        return self._sample_next_ordi()

    def _sample_next_ordi(self):

        if self.current_step == 0:
            # Reset the environment to an initial state sampled from p(s)
            s = torch.distributions.categorical.Categorical(probs=self.p_s).sample(self.batch_shape)
        else:
            # Sample next hidden state from current state and action p(s|s,a)
            s = torch.distributions.categorical.Categorical(probs=self.p_s_sa[self.last_s, self.a],).sample()

        # Sample observation, reward and done flag from p(o,r,d|s)
        if len(self.batch_shape) > 0:
            o = torch.distributions.categorical.Categorical(probs=self.p_o_s[s].view(*self.batch_shape, -1)).sample()
            r = torch.distributions.categorical.Categorical(probs=self.p_r_s[s].view(*self.batch_shape, -1)).sample()
            d = torch.distributions.bernoulli.Bernoulli(probs=self.p_d_s[s].view(*self.batch_shape)).sample()
        else:
            o = torch.distributions.categorical.Categorical(probs=self.p_o_s[s].view(-1)).sample()
            r = torch.distributions.categorical.Categorical(probs=self.p_r_s[s].view(-1)).sample()
            d = torch.distributions.bernoulli.Bernoulli(probs=self.p_d_s[s]).sample()

        d = d.to(dtype=o.dtype)  # float to long

        # make sure done flag is persistent (also clear observation and reward if done flag was previously raised)
        if self.current_step > 0:
            d = torch.max(self.last_d, d)
            o = torch.where(self.last_d > 0, 0, o)
            r = torch.where(self.last_d > 0, 0, r)
            s = torch.where(self.last_d > 0, 0, s)

        self.last_s = s
        self.last_d = d

        info = {"state": s}

        return o, r, d, info
