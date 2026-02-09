import pytest
import numpy as np
import sys
import gymnasium as gym
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.env_rl_gym import SimpleCPUEnv
from env.simple_cpu import OpCode

class TestSimpleCPUEnv:
    @pytest.fixture
    def env(self):
        return SimpleCPUEnv(target_value=42, max_cycles=20)

    def test_gym_api_compliance(self, env):
        """Test basic Gym API compliance."""
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        obs, info = env.reset()
        assert "registers" in obs
        assert "pc" in obs
        assert isinstance(obs["registers"], np.ndarray)
        assert obs["registers"].shape == (16,)
        assert isinstance(obs["pc"], int)

    def test_step_structure(self, env):
        """Test step return structure."""
        env.reset()
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_execution_updates_state(self, env):
        """Test that actions actually update the internal CPU state."""
        env.reset()
        
        # Action: MOV R8 <- R1 (Context: R1=1)
        # [MOV, R1, 0, R8, NextPC=1]
        # MOV is opcode 4
        action = np.array([4, 1, 0, 8, 1])
        
        obs, _, _, _, _ = env.step(action)
        
        # Check if R8 (index 8) is now 1
        assert obs["registers"][8] == 1
        assert obs["pc"] == 1
        assert env.cpu.registers[8] == 1

    def test_target_reward(self, env):
        """Test that reaching the target value yields a reward."""
        # Target is 42
        env.reset()
        
        # Force a value to 42 using internal CPU access for setup, 
        # or construct a sequence. Let's use internal setup for unit testing the reward logic itself.
        # But `step` checks the logic.
        
        # Let's do: MOV R8 <- R0 (0). Then manually set R8 to 42 to verify reward logic, 
        # OR simply specific instruction: 
        # Actually proper way:
        # R1=1. We need 42. It's hard to get 42 in one step from Reset.
        # So we can "cheat" by setting a register before the step.
        
        env.cpu._write_register(9, 42) # Set R9 to 42
        
        # Execute a dummy action
        action = np.array([OpCode.MOV, 9, 0, 10, 1]) # MOV R10 <- R9 (42)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Now 42 is in R9 (and R10).
        # Logic says: if target_value in registers[CONSTANT_REGISTERS:]
        # R9 and R10 are writable (>=8). So it should trigger.
        
        assert terminated == True
        assert reward > 0 # Should have positive reward

    def test_max_cycles_truncation(self, env):
        """Test that environment truncates after max_cycles."""
        env = SimpleCPUEnv(max_cycles=5)
        env.reset()
        
        terminated = False
        truncated = False
        
        # Run 5 steps
        for _ in range(5):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        assert truncated == True or terminated == True 
        # If it randomly hit target, it's terminated. If not, it's truncated at 5.
        # To be sure it truncates, we can ensure target is impossible or just check cycle count.
        assert env.cpu.cycle_count == 5

    def test_halt_termination(self, env):
        """Test HALT opcode terminates episode."""
        env.reset()
        
        # Action: HALT
        action = np.array([OpCode.HALT, 0, 0, 0, 0])
        
        _, _, terminated, _, _ = env.step(action)
        
        assert terminated == True
        assert env.cpu.halted == True
