from gymnasium import Wrapper

class ActionMaskWrapper(Wrapper):
    """
    This wrapper adds an 'action_masks' method to the environment,
    which returns the current legal moves mask as a Boolean array.
    """
    def action_masks(self):
        env = self.env

        # Unwrap the environment properly
        while hasattr(env, "env") and not hasattr(env, "get_action_mask"):
            env = env.env

        # Handle DummyVecEnv or SubprocVecEnv
        if hasattr(env, "envs"):  # If vectorized, access first env
            env = env.envs[0]

        # Now, we should have access to get_action_mask()
        return env.get_action_mask().astype(bool)
