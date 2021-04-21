from docplex.mp.model import Model
import numpy as np

# input file name is set as mat_raw.txt. remember to change it if you use a different file
tumor_file = "/Users/caron/PycharmProjects/ORProject/actualexample/tumor_raw.txt"
critical_file = "/Users/caron/PycharmProjects/ORProject/actualexample/critical_raw.txt"
beam_file = "/Users/caron/PycharmProjects/ORProject/actualexample/beam_raw.txt"


def convert_neighbors(input_array):
    '''Create a matrix where elements bordering 1s are 1s, if they are not tumors
  '''
    # create new zeros array to return
    array = np.array(input_array)[0]
    neighboring_array = np.zeros_like(array)
    array_shape = np.shape(array)
    max_edge_x = array_shape[0] - 1
    max_edge_y = array_shape[1] - 1

    for (x, y), val in np.ndenumerate(array):
        # [0] [1] [2]
        # [3] [*] [5]
        # [6] [7] [8]
        # * refers to current cell

        if (val == 1):
            if (x == 0):
                if (y != 0):
                    neighboring_array[x, y - 1] = 1 if array[x, y - 1] == 0 else 0
                    neighboring_array[x + 1, y - 1] = 1 if array[x + 1, y - 1] == 0 else 0
                    if (y != max_edge_y):
                        neighboring_array[x, y + 1] = 1 if array[x, y + 1] == 0 else 0
                        neighboring_array[x + 1, y + 1] = 1 if array[x + 1, y + 1] == 0 else 0
                neighboring_array[x + 1, y] = 1 if array[x + 1, y] == 0 else 0
            elif (x == max_edge_x):
                if (y != 0):
                    neighboring_array[x, y - 1] = 1 if array[x, y - 1] == 0 else 0
                    neighboring_array[x - 1, y - 1] = 1 if array[x - 1, y - 1] == 0 else 0
                    if (y != max_edge_y):
                        neighboring_array[x, y + 1] = 1 if array[x, y + 1] == 0 else 0
                        neighboring_array[x - 1, y + 1] = 1 if array[x - 1, y + 1] == 0 else 0
                neighboring_array[x - 1, y] = 1 if array[x - 1, y] == 0 else 0
            else:
                if (y != 0):
                    neighboring_array[x, y - 1] = 1 if array[x, y - 1] == 0 else 0
                    neighboring_array[x - 1, y - 1] = 1 if array[x - 1, y - 1] == 0 else 0
                    neighboring_array[x + 1, y - 1] = 1 if array[x + 1, y - 1] == 0 else 0
                    if (y != max_edge_y):
                        neighboring_array[x, y + 1] = 1 if array[x, y + 1] == 0 else 0
                        neighboring_array[x - 1, y + 1] = 1 if array[x - 1, y + 1] == 0 else 0
                        neighboring_array[x + 1, y + 1] = 1 if array[x + 1, y + 1] == 0 else 0
                neighboring_array[x - 1, y] = 1 if array[x - 1, y] == 0 else 0
                neighboring_array[x + 1, y] = 1 if array[x + 1, y] == 0 else 0
    return neighboring_array


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

        for i in range(self.num_matrices):
            if self.matrices[i] == []:
                self.matrices.pop(i)
                self.num_matrices -= 1
                break


def build_model(beam_matrices, tumor_matrix, critical_matrix, number_weeks,
                treatment_upper_bound, treatment_lower_bound, blood_toxic_upper_bound,
                treat_strength_diff, pen_tumor, pen_crit, pen_border):

    model = Model(log_output=True)
    # get the border of the tumor
    critical_border = convert_neighbors(critical_matrix.matrices)

    # Variables: x_i_j_k with k being the identifiers for the beam matrices

    r = model.continuous_var_matrix(keys1=beam.num_matrices, keys2=number_weeks, name="beam_strength")

    h = model.continuous_var_list(keys=number_weeks, name="chemo_amount")

    s_r = model.continuous_var_cube(keys1=beam.num_rows, keys2=beam.num_columns, keys3=number_weeks,
                                    name="critical_relax")

    s_t = model.continuous_var_cube(keys1=beam.num_rows, keys2=beam.num_columns, keys3=number_weeks,
                                    name="tumor_relax")

    # critical area constraint (radiation upper)
    for i in range(0, beam_matrices.num_rows):
        for j in range(0, beam_matrices.num_columns):
            for w in range(0, number_weeks):
                e = model.linear_expr()
                add = False
                for k in range(0, beam_matrices.num_matrices):
                    if critical_matrix.matrices[0][i][j] == 1 and beam_matrices.matrices[k][i][j] > 0:
                        e += (treat_strength_diff * r[(k, w)] * beam_matrices.matrices[k][i][j])
                        add = True
                e += h[w]
                if add:
                    model.add_constraint(e - s_r[(i, j, w)] <= treatment_upper_bound,
                                         ctname="critical_{0}{1}_{2}".format(i, j, w))


    # tumor area constraint (radiation lower)
    first_radiation = treatment_lower_bound

    for i in range(beam_matrices.num_rows):
        for j in range(beam_matrices.num_columns):
            treatment_lower_bound = first_radiation
            for w in range(0, number_weeks):
                e = model.linear_expr()
                add = False
                for k in range(beam_matrices.num_matrices):
                    if tumor_matrix.matrices[0][i][j] == 1 and beam_matrices.matrices[k][i][j] > 0:
                        e += (treat_strength_diff * r[(k, w)] * beam_matrices.matrices[k][i][j])
                        add = True
                e += h[w]
                if add:
                    model.add_constraint(e + s_t[(i, j, w)] >= treatment_lower_bound,
                                             ctname="tumor_{0}{1}_{2}".format(i, j, w))

                treatment_lower_bound = treatment_lower_bound - (first_radiation / number_weeks)

    first_blood_toxic = blood_toxic_upper_bound

    # blood toxicity upper bound update
    for w in range(0, number_weeks):
        for k in range(beam_matrices.num_matrices):
            model.add_constraint(r[(k, w)] <= blood_toxic_upper_bound, ctname="blood_tox_{0}_{1}".format(w,k))
        blood_toxic_upper_bound = blood_toxic_upper_bound - (first_blood_toxic / number_weeks)


    # non-negativity constraints for chemo and radiation
    for w in range(0, number_weeks):
        model.add_constraint(h[w] >= 0)
        for k in range(0, beam_matrices.num_matrices):
            model.add_constraint(r[(k, w)] >= 0)

    # create objective function
    e = model.linear_expr()

    for i in range(beam_matrices.num_rows):
        for j in range(beam_matrices.num_columns):
            for w in range(number_weeks):
                for k in range(beam_matrices.num_matrices):
                    e += r[(k, w)] * beam_matrices.matrices[k][i][j] * critical_matrix.matrices[0][i][j] * treat_strength_diff  # sum of beams on each critical cell
                    e += r[(k, w)] * beam_matrices.matrices[k][i][j] * critical_border[i][j] * treat_strength_diff / pen_border  # sum of beams on the border

                e += pen_tumor * s_t[(i, j, w)]  # slack for tumor
                e += pen_crit * s_r[(i, j, w)]  # slack for critical area
                e += h[w] * critical_matrix.matrices[0][i][j]  # chemo on critical area
                e += h[w] * critical_border[i][j] / pen_border  # chemo on border

    model.minimize(e)

    return model


def print_result(filename, vars, data, number_weeks):
    file = open(filename, 'w')
    for i in range(data.num_rows):
        for j in range(data.num_columns):
            out = 0
            for k in range(data.num_matrices):
                for w in range(number_weeks):
                    out += vars[k + w * data.num_matrices].solution_value * data.matrices[k][i][j]
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
                          number_weeks=12, treatment_upper_bound=4, treatment_lower_bound=9,
                          blood_toxic_upper_bound=4, treat_strength_diff=3, pen_tumor=50,
                          pen_crit=100, pen_border=2)
    s = model_1.solve()
    print("Objective value is {0}".format(s.get_objective_value()))

    for var, value in s.iter_var_values():
        print("Variable {0} is equal to {1}".format(var, value))

    #for constraint in model_1.iter_constraints():
        #print(constraint)

    x_vars = model_1.find_matching_vars(pattern="beam_strength_")
    print_result("p4_results.out", x_vars, beam, 12)

    #Collects the constraint types
    cpx = model_1.get_engine().get_cplex()
    crit_cts = model_1.find_matching_linear_constraints(pattern="critical_")
    tum_cts = model_1.find_matching_linear_constraints(pattern="tumor_")
    bt_cts = model_1.find_matching_linear_constraints(pattern="blood_tox_")

    #Sensitivity analysis on the different constraint types
    crit_const_sens = []
    tum_const_sens = []
    rem_const_sens = []
    for i in range(len(crit_cts)):
        crit_const_sens.append(cpx.solution.sensitivity.rhs(i))
    for i in range(len(crit_cts),(len(crit_cts)+len(tum_cts)),12):
        tum_const_sens.append(cpx.solution.sensitivity.rhs(i))
    for i in range((len(crit_cts)+len(tum_cts)),(len(crit_cts)+len(tum_cts)+125)):
        rem_const_sens.append(cpx.solution.sensitivity.rhs(i))

    # Prints the allowable ranges for the critical upper bound, tumor lower bound, and blood toxicity upper bound
    print("----------------------")
    print("The current solution is feasible for critical area upper bound values "
          "between: {0} and {1}".format(min(crit_const_sens)[1], max(crit_const_sens)[0]))
    print("The current solution is feasible for week 1 tumor area lower bound values "
          "between: {0} and {1}".format(min(tum_const_sens)[1], max(tum_const_sens)[0]))
    print("The current solution is feasible for week 1 blood toxicity upper bound values "
          "between: {0} and {1}".format(min(rem_const_sens)[1], max(rem_const_sens)[0]))
    print("----------------------")
