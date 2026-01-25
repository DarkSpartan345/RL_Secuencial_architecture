"""
test_gym_env.py - Pruebas unitarias para el entorno Gymnasium
Verifica que el entorno cumpla con las especificaciones de Gymnasium
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.gym_env import MultiplicationEnv


class TestEnvironmentBasics:
    """Pruebas básicas del entorno"""
    
    def test_environment_creation(self):
        """Verifica que el entorno se cree correctamente"""
        env = MultiplicationEnv(bit_width=8, max_cycles=32)
        
        assert env.bit_width == 8
        assert env.max_cycles == 32
        assert env.num_states == 16
    
    def test_observation_space(self):
        """Verifica el espacio de observación"""
        env = MultiplicationEnv(bit_width=8)
        
        assert env.observation_space.shape == (7,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)
    
    def test_action_space(self):
        """Verifica el espacio de acciones"""
        env = MultiplicationEnv()
        
        # Verificar que sea MultiDiscrete con las dimensiones correctas
        assert hasattr(env.action_space, 'nvec')
        expected_nvec = [7, 3, 4, 5, 4, 2, 2, 2, 16]
        assert list(env.action_space.nvec) == expected_nvec


class TestResetFunctionality:
    """Pruebas de la función reset"""
    
    def test_reset_returns_observation_and_info(self):
        """Verifica que reset retorne observación e info"""
        env = MultiplicationEnv(bit_width=8)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (7,)
        assert isinstance(info, dict)
    
    def test_reset_with_seed(self):
        """Verifica que el seed funcione correctamente"""
        env1 = MultiplicationEnv(bit_width=8)
        env2 = MultiplicationEnv(bit_width=8)
        
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)
        
        # Deberían generar los mismos valores
        assert info1['a'] == info2['a']
        assert info1['b'] == info2['b']
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_reset_with_custom_values(self):
        """Verifica reset con valores personalizados"""
        env = MultiplicationEnv(bit_width=8)
        obs, info = env.reset(options={'a': 5, 'b': 7})
        
        assert info['a'] == 5
        assert info['b'] == 7
        assert info['expected'] == 35
    
    def test_reset_initializes_state(self):
        """Verifica que reset inicialice el estado correctamente"""
        env = MultiplicationEnv(bit_width=8)
        
        # Hacer algunos pasos
        env.reset()
        action = env.action_space.sample()
        env.step(action)
        env.step(action)
        
        # Resetear
        obs, info = env.reset()
        
        assert env.pc == 0
        assert env.done == False
        assert env.datapath.cycle_count == 0


class TestStepFunctionality:
    """Pruebas de la función step"""
    
    def test_step_returns_correct_tuple(self):
        """Verifica que step retorne la tupla correcta"""
        env = MultiplicationEnv(bit_width=8)
        env.reset()
        
        action = env.action_space.sample()
        result = env.step(action)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_updates_cycle_count(self):
        """Verifica que step incremente el contador de ciclos"""
        env = MultiplicationEnv(bit_width=8)
        env.reset()
        
        action = env.action_space.sample()
        _, _, _, _, info1 = env.step(action)
        _, _, _, _, info2 = env.step(action)
        
        assert info2['cycle_count'] == info1['cycle_count'] + 1
    
    def test_step_updates_pc(self):
        """Verifica que step actualice el program counter"""
        env = MultiplicationEnv(bit_width=8)
        env.reset()
        
        # Acción que va al estado 5
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 5])
        _, _, _, _, info = env.step(action)
        
        assert info['pc'] == 5
    
    def test_step_after_done_raises_error(self):
        """Verifica que step después de done lance error"""
        env = MultiplicationEnv(bit_width=8, max_cycles=2)
        env.reset()
        
        action = env.action_space.sample()
        env.step(action)
        env.step(action)  # Esto debería truncar
        
        with pytest.raises(RuntimeError):
            env.step(action)


class TestTerminationConditions:
    """Pruebas de condiciones de terminación"""
    
    def test_termination_on_halt_state(self):
        """Verifica terminación al llegar al estado HALT (15)"""
        env = MultiplicationEnv(bit_width=8)
        env.reset(options={'a': 5, 'b': 3})
        
        # Configurar el resultado correcto manualmente
        env.datapath.reg_p = 15
        
        # Acción que va al estado 15 (HALT)
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 15])
        _, reward, terminated, truncated, info = env.step(action)
        
        assert terminated == True
        assert truncated == False
        # Debería tener bonus por resultado correcto
        assert reward > 0
    
    def test_truncation_on_max_cycles(self):
        """Verifica truncamiento al exceder ciclos máximos"""
        env = MultiplicationEnv(bit_width=8, max_cycles=3)
        env.reset()
        
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  # No va a HALT
        
        env.step(action)
        env.step(action)
        _, reward, terminated, truncated, info = env.step(action)
        
        assert truncated == True
        assert terminated == False
        assert info['cycle_count'] >= 3
    
    def test_success_bonus_on_correct_result(self):
        """Verifica bonus por resultado correcto"""
        env = MultiplicationEnv(bit_width=8)
        env.reset(options={'a': 4, 'b': 5})
        
        # Establecer resultado correcto
        env.datapath.reg_p = 20
        
        # Ir a HALT
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 15])
        _, reward, terminated, _, info = env.step(action)
        
        assert terminated == True
        assert info['correct'] == True
        # El reward debería incluir success_bonus
        assert reward >= env.reward_calc.success_bonus


class TestObservationNormalization:
    """Pruebas de normalización de observaciones"""
    
    def test_observation_is_normalized(self):
        """Verifica que las observaciones estén normalizadas [0, 1]"""
        env = MultiplicationEnv(bit_width=8)
        obs, _ = env.reset(options={'a': 100, 'b': 200})
        
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
    
    def test_observation_reflects_state(self):
        """Verifica que la observación refleje el estado del datapath"""
        env = MultiplicationEnv(bit_width=8)
        obs, info = env.reset(options={'a': 128, 'b': 64})
        
        # obs = [reg_a, reg_b, reg_p, reg_temp, flag_z, flag_lsb, flag_c]
        assert obs[0] == pytest.approx(128 / 255)  # reg_a normalizado
        assert obs[1] == pytest.approx(64 / 255)   # reg_b normalizado
        assert obs[2] == 0.0  # reg_p empieza en 0
        assert obs[5] == 0.0  # 64 es par, LSB=0


class TestActionDecoding:
    """Pruebas de decodificación de acciones"""
    
    def test_action_decoding(self):
        """Verifica que las acciones se decodifiquen correctamente"""
        env = MultiplicationEnv(bit_width=8)
        env.reset()
        
        action = np.array([0, 1, 2, 3, 1, 1, 0, 1, 5])
        env.step(action)
        
        # Verificar que la acción se aplicó (PC debería ser 5)
        assert env.pc == 5


class TestInfoDictionary:
    """Pruebas del diccionario de información"""
    
    def test_info_contains_required_fields(self):
        """Verifica que info contenga los campos necesarios"""
        env = MultiplicationEnv(bit_width=8)
        _, info = env.reset()
        
        required_fields = ['pc', 'cycle_count', 'a', 'b', 'expected', 'actual', 'correct']
        for field in required_fields:
            assert field in info
    
    def test_info_correctness_flag(self):
        """Verifica que el flag 'correct' funcione"""
        env = MultiplicationEnv(bit_width=8)
        _, info = env.reset(options={'a': 6, 'b': 7})
        
        # Al inicio, no es correcto
        assert info['correct'] == False
        
        # Establecer resultado correcto manualmente
        env.datapath.reg_p = 42
        _, _, _, _, info = env.step(env.action_space.sample())
        
        # Ahora debería ser correcto
        assert info['correct'] == True


class TestIntegrationScenarios:
    """Pruebas de escenarios de integración"""
    
    def test_simple_episode(self):
        """Ejecuta un episodio simple completo"""
        env = MultiplicationEnv(bit_width=8, max_cycles=10)
        obs, info = env.reset(seed=42)
        
        done = False
        steps = 0
        
        while not done and steps < 10:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps <= 10
        assert done == True
    
    def test_multiple_episodes(self):
        """Ejecuta múltiples episodios"""
        env = MultiplicationEnv(bit_width=8, max_cycles=5)
        
        for episode in range(3):
            obs, info = env.reset()
            
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
    
    def test_deterministic_execution(self):
        """Verifica que la ejecución sea determinística con mismas acciones"""
        env1 = MultiplicationEnv(bit_width=8)
        env2 = MultiplicationEnv(bit_width=8)
        
        env1.reset(seed=123, options={'a': 10, 'b': 20})
        env2.reset(seed=123, options={'a': 10, 'b': 20})
        
        action = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])
        
        obs1, r1, t1, tr1, info1 = env1.step(action)
        obs2, r2, t2, tr2, info2 = env2.step(action)
        
        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
