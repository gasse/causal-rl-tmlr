import torch
from utils import print_log

class UniformAgent(torch.nn.Module):

    """ Uniform policy to act within a POMDP environment with random actions """

    def __init__(self, a_nvals):
        super().__init__()
        self.a_nvals = a_nvals

    def reset(self):
        pass

    def action(self, *args, **kwargs):

        action = torch.distributions.categorical.Categorical(
            probs=torch.ones(self.a_nvals)/self.a_nvals).sample()
 
        return action

class PrivilegedAgent(torch.nn.Module):

    """ Expert Policy Class that chooses its actions from the hidden state s """

    def __init__(self, p_a_s):
        super().__init__()
        self.probs_a_s = torch.as_tensor(p_a_s)  # p(a_t | s_t, i=0)

    def reset(self):
        pass

    def action(self, state):

        action = torch.distributions.categorical.Categorical(probs=self.probs_a_s[state]).sample()

        return action

class EpsilonRandomAgent(torch.nn.Module):

    def __init__(self, base_policy, a_nvals, epsilon):
        super().__init__()
        assert epsilon >= 0 and epsilon <= 1

        self.override_probs = torch.nn.Parameter(torch.tensor(epsilon).to(dtype=torch.float))
        self.random_policy_probs = torch.nn.Parameter(torch.ones(a_nvals).to(dtype=torch.float)/a_nvals)

        self.base_policy = base_policy

    def reset(self):
        self.base_policy.reset()

    def action(self, *args, **kwargs):

        # always query base policy (statefull)
        base_action = self.base_policy.action(*args, **kwargs)

        batch_size = base_action.size()

        override = torch.distributions.bernoulli.Bernoulli(probs=self.override_probs).sample(batch_size)
        random_action = torch.distributions.categorical.Categorical(probs=self.random_policy_probs).sample(batch_size)

        action = torch.where(override == 1, random_action, base_action)

        return action

class BeliefStateActorCriticAgent(torch.nn.Module):

    def __init__(self, belief_model, hidden_size):
        super().__init__()
        self.belief_model = belief_model
        self.hidden_size = hidden_size

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(belief_model.s_nvals, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, belief_model.a_nvals),
            torch.nn.LogSoftmax(dim=-1),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(belief_model.s_nvals, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def reset(self):
        # reset agent's last belief state and action (agent memory)
        self.last_belief_state = None
        self.last_action = None

    def action(self, obs, reward, done, training=False):

        assert self.last_action is None or self.last_action.size() == reward.size()

        # 1. update agent's belief state using the model
        last_log_q_s_h = None if self.last_belief_state is None else self.last_belief_state
        last_a = None if self.last_action is None else self.last_action
        o = obs
        r = reward
        d = done

        batched = len(reward.size()) > 0

        if not batched:
            last_log_q_s_h = None if last_log_q_s_h is None else last_log_q_s_h.unsqueeze(0)
            last_a = None if last_a is None else last_a.unsqueeze(0)
            o = o.unsqueeze(0)
            r = r.unsqueeze(0)
            d = d.unsqueeze(0)

        with torch.no_grad():
            belief_state = self.belief_model.log_q_s_h(
                regime=torch.ones_like(reward),  # interventional regime
                loq_q_sprev_hprevi=last_log_q_s_h, a=last_a, o=o, r=r, d=d)

        if not batched:
            belief_state = belief_state.squeeze(0)


        # 2. obtain agent's action
        with torch.set_grad_enabled(training):
            policy_log_probs = self.actor(torch.exp(belief_state))
            action = torch.distributions.categorical.Categorical(logits=policy_log_probs).sample()

        # 3. obtain belief state's value estimate (critic), if needed for training
        if training:
            belief_state_value = self.critic(torch.exp(belief_state)).squeeze(-1)

        # store agent's last belief state and action (agent memory)
        self.last_belief_state = belief_state
        self.last_action = action

        if training:
            return action, policy_log_probs, belief_state_value
        else:
            return action


def train_ac_agent(env, agent, reward_map, max_episode_length,
                gamma=1,
                n_epochs_warmup=10,
                n_epochs=100,
                epoch_size=20,
                batch_size=4,
                critic_weight=1,
                entropy_bonus=0,
                lr=1e-2,
                scale_returns=False,
                log=False,
                logfile=None):

    # infer the device from the agent
    device = next(agent.parameters()).device

    reward_map = torch.as_tensor(reward_map, device=device, dtype=torch.float)

    assert reward_map.size() == torch.Size((env.r_nvals, ))

    if log:
        print_log(f"AC n_epochs: {n_epochs}", logfile)
        print_log(f"AC n_epochs_warmup: {n_epochs_warmup}", logfile)
        print_log(f"AC epoch_size: {epoch_size}", logfile)
        print_log(f"AC batch_size: {batch_size}", logfile)
        print_log(f"AC critic_weight: {critic_weight}", logfile)
        print_log(f"AC lr: {lr}", logfile)
        print_log(f"AC device: {device}", logfile)

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    for ep in range(n_epochs + n_epochs_warmup + 1) :

        epoch_return = 0
        epoch_actor_loss = 0
        epoch_actor_loss_abs = 0
        epoch_critic_loss = 0
        epoch_actor_entropy = 0

        if log and ep == 0 :
            print_log(f"Initial evaluation epoch (no training)", logfile=logfile)

        if log and ep == 1 :
            print_log(f"AC critic pre-training for {n_epochs_warmup} epochs", logfile=logfile)

        if log and ep == n_epochs_warmup + 1:
            print_log(f"AC actor-critic training for {n_epochs} epochs", logfile=logfile)

        for _ in range(epoch_size):

            loss = 0.

            values, rewards, policy_entropies, action_log_probs, seq_was_done = [], [], [], [], []

            # run one batch of episodes
            agent.reset()
            with torch.no_grad():
                obs, reward, done, _ = env.reset(batch_size)

            was_done = done
            t = 0

            while True:

                # stop if maximum episode length reached
                if t == max_episode_length:
                    break

                # early stop if all episodes have ended
                if all(done):
                    break

                action, policy_log_probs, value = agent.action(obs, reward, done, training=True)
                with torch.no_grad():
                    obs, reward, done, _ = env.step(action)

                values.append(value)
                rewards.append(torch.index_select(reward_map, 0, reward))  # convert discrete rewards to their numerical values
                policy_entropies.append(torch.distributions.categorical.Categorical(logits=policy_log_probs).entropy())
                action_log_probs.append(policy_log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))
                seq_was_done.append(was_done)

                # enforce persistent done flag (should be true)
                was_done = torch.max(done, was_done)
                t += 1

            values = torch.stack(values)
            rewards = torch.stack(rewards)
            policy_entropies = torch.stack(policy_entropies)
            action_log_probs = torch.stack(action_log_probs)
            seq_was_done = torch.stack(seq_was_done)

            # discard transitions after done flag raised (due to batch padding)
            values = torch.where(seq_was_done==1, 0, values)
            rewards = torch.where(seq_was_done==1, 0, rewards)
            policy_entropies = torch.where(seq_was_done==1, 0, policy_entropies)
            action_log_probs = torch.where(seq_was_done==1, 0, action_log_probs)

            n_transitions = (1 - seq_was_done).sum()

            # compute (discounted) returns from rewards
            returns = torch.empty_like(rewards)
            for i in reversed(range(rewards.size(0))):
                if i == rewards.size(0)-1:
                    returns[i] = rewards[i]
                else:
                    returns[i] = rewards[i] + gamma * returns[i+1]

            # adjust return signals using critic baseline
            return_signals = (returns - values.detach())

            # scale return signals (std=1), if asked
            if scale_returns:
                returns_mean = returns.sum() / n_transitions
                if n_transitions > 1:
                    returns_std = torch.sqrt(torch.sum(torch.where(seq_was_done==1, 0, returns-returns_mean) ** 2)) / (n_transitions - 1)
                if n_transitions <= 1 or returns_std == 0.0:
                    returns_std = returns.abs().max()
                return_signals /= returns_std + 1e-4

            # compute actor-critic loss values
            actor_loss = torch.sum(-action_log_probs * return_signals) / n_transitions
            # critic_loss =  torch.nn.functional.mse_loss(values, returns, reduction = 'sum') / n_transitions
            critic_loss =  torch.nn.functional.smooth_l1_loss(values, returns, reduction = 'sum') / n_transitions
            actor_entropy = policy_entropies.sum() / n_transitions

            loss += critic_weight * critic_loss

            if ep >= n_epochs_warmup:
                loss += actor_loss - entropy_bonus * actor_entropy

            epoch_return += rewards.sum().item() / batch_size
            epoch_actor_loss += actor_loss.detach().item()
            epoch_actor_loss_abs += (torch.sum(-action_log_probs * return_signals.abs()) / n_transitions).detach().item()
            epoch_critic_loss += critic_loss.detach().item()
            epoch_actor_entropy += actor_entropy.detach().item()

            if ep > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_return /= epoch_size
        epoch_actor_loss /= epoch_size
        epoch_actor_loss_abs /= epoch_size
        epoch_critic_loss /= epoch_size
        epoch_actor_entropy /= epoch_size

        if log:
            print_log(f"AC epoch {ep:04d} / {n_epochs + n_epochs_warmup:04d} return={epoch_return:0.4f} actor_loss={epoch_actor_loss:+0.4f} actor_loss_abs={epoch_actor_loss_abs:+0.4f} critic_loss={epoch_critic_loss:0.4f} actor_entropy={epoch_actor_entropy:0.4f}", logfile=logfile)


def evaluate_agent(env, agent, reward_map, max_episode_length, n_samples, batch_size):

    # infer the device from the agent
    device = next(agent.parameters()).device

    reward_map = torch.as_tensor(reward_map, device=device, dtype=torch.float)

    avg_return = 0.
    n_samples_done = 0

    while n_samples_done < n_samples:
        n_samples_todo = min(batch_size, n_samples - n_samples_done)

        with torch.no_grad():

            agent.reset()
            obs, reward, done, _ = env.reset(n_samples_todo)

            # avg_return += torch.index_select(reward_map, 0, reward).sum().item()  # discard initial reward (arbitrary choice)

            was_done = done.to(dtype=torch.bool)
            t = 0

            while True:

                # stop if maximum episode length reached
                if t == max_episode_length:
                    break

                # early stop if all episodes have ended
                if all(done):
                    break

                action = agent.action(obs, reward, done)
                obs, reward, done, _ = env.step(action)

                avg_return += torch.where(was_done, 0, torch.index_select(reward_map, 0, reward)).sum().item()  # enforce rewards equal to 0 after done flag raised (should be true) + convert discrete rewards to their numerical values

                was_done = torch.logical_or(done, was_done)  # enforce persistent done flag after raised (should be true)
                t += 1

        n_samples_done += n_samples_todo

    avg_return /= n_samples

    return avg_return