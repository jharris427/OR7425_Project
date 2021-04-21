from docplex.mp.model import Model

# input file name is set as mat_raw.txt. remember to change it if you use a different file
class modeling_data():
    def __init__(self):
        self.data_file_name = "mat_raw.txt"

        self.num_matrices = None
        self.num_rows = None
        self.num_columns = None

        self.matrices = [[]]

    def get_data_from_file(self):
        file = open(self.data_file_name, 'r')
        lines = file.readlines()
        for line in lines:
            l = line[:-1].split()
            if l:
                self.matrices[-1].append([])
                for i in l:
                    self.matrices[-1][-1].append(int(i))
            else:
                self.matrices.append([])

        self.num_matrices = len(self.matrices)
        self.num_rows = len(self.matrices[0])
        self.num_columns = len(self.matrices[0][0])

def build_model(data):
    model = Model(log_output=True)

    # Variables: x_i_j is the maximum value in the matrices
    x = model.continuous_var_matrix(keys1=data.num_rows, keys2=data.num_columns, name="x")

    for i in range(data.num_rows):
        for j in range(data.num_columns):
            for k in range(data.num_matrices):
                model.add_constraint(x[i,j] >= data.matrices[k][i][j])

    e = model.linear_expr()
    for i in range(data.num_rows):
        for j in range(data.num_columns):
            e += x[i,j]

    model.minimize(e)

    return model

def print_result(filename, vars, data):
    file = open(filename, 'w')
    for i in range(data.num_rows):
        for j in range(data.num_columns):
            file.write(str(vars[i * data.num_rows + j].solution_value) + '\t')
        file.write('\n')


data = modeling_data()
data.get_data_from_file()
model = build_model(data)

s = model.solve()
x_vars = model.find_matching_vars(pattern="x_")
print_result("results.out", x_vars, data)
