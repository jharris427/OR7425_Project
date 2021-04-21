from docplex.mp.model import Model
import numpy as np

# input file name is set as mat_raw.txt. remember to change it if you use a different file
tumor_file = "/Users/Sam/PycharmProjects/ORProject/smallexample/tumor_raw.txt"
critical_file = "/Users/Sam/PycharmProjects/ORProject/smallexample/critical_raw.txt"
beam_file = "/Users/Sam/PycharmProjects/ORProject/smallexample/beam_raw.txt"

class modeling_data():

    def __init__(self):
        self.num_matrices = None
        self.num_rows = None
        self.num_columns = None
        self.matrices = [[]]

    def get_data_from_file(self, file_name):
        data_file_name = file_name
        file = open(data_file_name, 'r')
        lines = file.readlines()
        for line in lines:
            l = line[:-1].split()
            if l:
                self.matrices[-1].append([])
                for i in l:
                    self.matrices[-1][-1].append(float(i))
            else:
                self.matrices.append([])

        self.num_matrices = len(self.matrices)
        self.num_rows = len(self.matrices[0])
        self.num_columns = len(self.matrices[0][0])


        # get rid of any extra matrices if they are read in
        for i in range(self.num_matrices):
            if self.matrices[i] == []:
                self.matrices.pop(i)
                self.num_matrices -= 1
                break

def build_model(beam_matrices, tumor_matrix, critical_matrix, treatment_upper_bound, treatment_lower_bound):
    model = Model(log_output=True)
    # Variables: x_i_j_k with k being the identifiers for the beam matrices
    x = model.continuous_var_list(keys=beam_matrices.num_matrices, name="beam_strength")

    # critical area constraint <= 2
    for i in range(0, beam_matrices.num_rows):
        for j in range(0, beam_matrices.num_columns):
            e = model.linear_expr()
            add = False
            for k in range(0, beam_matrices.num_matrices):
                if critical_matrix.matrices[0][i][j] == 1 and beam_matrices.matrices[k][i][j] > 0:
                    e += (x[k] * beam_matrices.matrices[k][i][j])
                    add = True
            if add:
                model.add_constraint(e <= treatment_upper_bound)
                print(e)

    # tumor area constraint >=10

    for i in range(beam_matrices.num_rows):
        for j in range(beam_matrices.num_columns):
            e = model.linear_expr()
            add = False
            for k in range(beam_matrices.num_matrices):
                if tumor_matrix.matrices[0][i][j] == 1 and beam_matrices.matrices[k][i][j] > 0:
                    e += (x[k] * beam_matrices.matrices[k][i][j])
                    add = True
            if add:
                model.add_constraint(e >= treatment_lower_bound)
                print(e)

    #non-negativity constraints
    for k in range(0, beam_matrices.num_matrices):
        model.add_constraint(r[(k, w)] >= 0)

    # objective function
    e = model.linear_expr()
    for i in range(beam_matrices.num_rows):
        for j in range(beam_matrices.num_columns):
            for k in range(beam_matrices.num_matrices):
                e += x[k] * beam_matrices.matrices[k][i][j] * critical_matrix.matrices[0][i][j]

    model.minimize(e)

    return model


def print_result(filename, vars, data):
    file = open(filename, 'w')
    for i in range(data.num_rows):
        out = 0
        for j in range(data.num_columns):
            for k in range(data.num_matrices):
                out += vars[k].solution_value * data.matrices[k][i][j]
            file.write(str(out) + '\t')
        file.write('\n')


tumor = modeling_data()
tumor.get_data_from_file(tumor_file)

critical = modeling_data()
critical.get_data_from_file(critical_file)

beam = modeling_data()
beam.get_data_from_file(beam_file)

if True == True:

    model_1 = build_model(beam_matrices=beam, tumor_matrix=tumor, critical_matrix=critical,
                          treatment_upper_bound=2, treatment_lower_bound=10)
    s = model_1.solve()
    print("Objective value is {0}".format(s.get_objective_value()))
    for var, value in s.iter_var_values():
        print("Variable {0} is equal to {1}".format(var, value))

    x_vars = model_1.find_matching_vars(pattern="beam_strength_")
    print_result("p1_results_small.out", x_vars, beam)
