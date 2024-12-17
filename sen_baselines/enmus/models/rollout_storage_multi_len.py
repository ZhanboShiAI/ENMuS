from collections import defaultdict
import torch
import numpy as np

from ss_baselines.savi.models.rollout_storage import RolloutStorage, ExternalMemory


class RolloutStorageMultiLen(RolloutStorage):
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        use_external_memory,
        external_memory_size,
        external_memory_capacity,
        external_memory_dim,
        num_recurrent_layers=1,
    ):
        super().__init__(
            num_steps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            use_external_memory,
            external_memory_size,
            external_memory_capacity,
            external_memory_dim,
            num_recurrent_layers,
        )

        if use_external_memory:
            self.em = ExternalMemoryMultiLen(
                num_envs, self.em_size, self.em_capacity,
                self.em_dim, num_copies=num_steps + 1
            )
        else:
            self.em = None

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        not_done_masks,
        em_features,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(not_done_masks)
        if self.use_external_memory:
            self.em.insert(em_features, not_done_masks, self.step+1)
            self.em_masks[self.step + 1].copy_(self.em.masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        if self.use_external_memory:
            self.em_masks[0].copy_(self.em_masks[self.step])
            self.em.step2idx[0] = self.em.step2idx[self.step]
        self.step = 0

    def external_memory(self, step):
        memory_list = self.em.memory[:, step]
        if self.em.is_mapping:
            step = self.em.step2idx[step]
        
        if step >= self.em.capacity:
            memory = memory_list[step-self.em.capacity:step]
        else:
            memory = torch.cat([memory_list[step-self.em.capacity:], memory_list[:step]], dim=0)
        # format the memory to [memory_size, batch, feat_dim] and in sequence
        return memory.contiguous()

    
    def external_memory_masks(self, step):
        masks_list = self.em_masks[step]
        
        if self.em.is_mapping:
            step = self.em.step2idx[step]
        
        if step >= self.em.capacity:
            masks = masks_list[:, step-self.em.capacity:step]
        else:
            masks = torch.cat([masks_list[:, step-self.em.capacity:], masks_list[:, :step]], dim=1)
        return masks.contiguous()

    
    def external_memory_idx(self):
        return self.em.idx
    

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            if self.use_external_memory:
                em_store_batch = []
                em_masks_batch = []
            else:
                em_store_batch = None
                em_masks_batch = None

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                adv_targ.append(advantages[: self.step, ind])
                if self.use_external_memory:
                    temp_memory_list = []
                    temp_memory_masks_list = []
                    for step in range(self.step):
                        temp_memory = self.external_memory(step).unsqueeze(1)
                        temp_memory_list.append(temp_memory)
                        tmep_masks = self.external_memory_masks(step).unsqueeze(0)
                        temp_memory_masks_list.append(tmep_masks)
                    
                    em_store = torch.cat(temp_memory_list, dim=1)[:, :, ind]
                    em_masks = torch.cat(temp_memory_masks_list, dim=0)[:, ind]

                    em_store_batch.append(em_store)
                    em_masks_batch.append(em_masks)

            T, N = self.step, num_envs_per_batch

            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)
            if self.use_external_memory:
                em_store_batch = torch.stack(em_store_batch, 2)
                em_masks_batch = torch.stack(em_masks_batch, 1)

            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)
            if self.use_external_memory:
                em_store_batch = em_store_batch.view(-1, T * N, self.em_dim)
                em_masks_batch = self._flatten_helper(T, N, em_masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                em_store_batch,
                em_masks_batch,
            )

    
class ExternalMemoryMultiLen(ExternalMemory):
    def __init__(self, num_envs, total_size, capacity, dim, num_copies=1, is_mapping=True):
        super().__init__(num_envs, total_size, capacity, dim, num_copies)
        self.step2idx = np.zeros((total_size-capacity+1, ), dtype=int)
        self.is_mapping = is_mapping

    def insert(self, em_features, not_done_masks, step=0):
        self.memory[self.idx].copy_(em_features.unsqueeze(0))
        # Account for overflow capacity
        capacity_overflow_flag = self.masks.sum(1) == self.capacity
        assert(not torch.any(self.masks.sum(1) > self.capacity))
        self.masks[capacity_overflow_flag, self.idx - self.capacity] = 0.0
        self.masks[:, self.idx] = 1.0
        # Mask out the entire memory for the next observation if episode done
        self.masks *= not_done_masks

        if self.is_mapping:
            self.step2idx[step] = (self.idx + 1) % self.total_size

        self.idx = (self.idx + 1) % self.total_size