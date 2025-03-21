import math

def validate(fpath):
    with open(fpath, 'r') as file:
        i = 0
        for line in file:
            parts = line.strip().split()
            
            task_id = int(parts[0])
            task_name = parts[1]
            result = float(parts[2])
            x = float(parts[3]) if len(parts) > 3 else None
            y = float(parts[4]) if len(parts) > 4 else None
            
            if task_name == "Sin":
                expected = math.sin(x)
            elif task_name == "Sqrt":
                expected = math.sqrt(x)
            elif task_name == "Pow":
                expected = math.pow(x, y)
            
            if abs(result - expected) > 0.001:
                print(f"Test failed at line {task_id}: Expected {expected}, Got {result}")
                return
            i += 1
    
    print(f"All {i} tests passed")


if __name__ == "__main__":
    output_file = "/home/a.tishkin1/Theory_of_parallelism/lab3/sub2/build/output.txt"
    validate(output_file)
