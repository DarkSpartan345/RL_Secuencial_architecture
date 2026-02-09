import numpy as np
from env_rl_gym import SimpleCPUEnv  # Asegúrate de que el nombre del archivo coincida

def run_test_suite():
    # Inicializamos el entorno en modo humano para ver los prints
    env = SimpleCPUEnv(target_value=100, max_cycles=50, render_mode="human")
    env.reset()
    
    print("=== INICIANDO TEST DE ARQUITECTURA SIMPLECPU ===")

    # Formato de acción: [OpCode, Reg_A, Reg_B, Dest_1, Dest_2]
    # OpCodes sugeridos: 0:ADD, 1:SUB, 2:SHL, 3:SHR, 4:MOV, 5:BEQ, 6:BGT, 7:HALT
    
    tests = [
        # 1. Probar ADD: R1(1) + R2(2) = 3 -> Guardar en R08
        {"name": "Suma Simple (1+2=3)", "action": [0, 1, 2, 8, 10]},
        
        # 2. Probar MOV y SUB: Mover R8 a R9, luego Restar R9 - R1 = 2 -> Guardar en R10
        {"name": "Movimiento R8->R9", "action": [4, 8, 0, 9, 11]},
        {"name": "Resta (3-1=2)", "action": [1, 9, 1, 10, 12]},
        
        # 3. Probar Protección de Constantes: Intentar escribir en R0 (debe fallar)
        {"name": "Protección R0 (Escribir en constante)", "action": [4, 4, 0, 0, 13]},
        
        # 4. Probar Operaciones de Bits: SHL (2 << 2 = 8) -> Guardar en R11
        {"name": "Shift Left (2 << 2)", "action": [2, 10, 2, 11, 14]},
        
        # 5. Probar Saltos Condicionales (BGT): Si R11(8) > R10(2) saltar a PC 25
        {"name": "Salto Condicional BGT (True)", "action": [6, 11, 10, 25, 15]},
        
        # 6. Probar Salto Condicional (BEQ): Si R0(0) == R10(2) (False) ir a PC 5
        {"name": "Salto Condicional BEQ (False)", "action": [5, 0, 10, 99, 5]},
        
        # 7. HALT
        {"name": "Instrucción de Parada", "action": [7, 0, 0, 0, 0]}
    ]

    for i, test in enumerate(tests):
        print(f"\nTEST #{i+1}: {test['name']}")
        action = np.array(test['action'])
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated and i < len(tests) - 1 and test['action'][0] != 7:
            print("¡AVISO: El entorno terminó antes de tiempo!")
            break

    print("\n=== PRUEBAS FINALIZADAS ===")

if __name__ == "__main__":
    run_test_suite()