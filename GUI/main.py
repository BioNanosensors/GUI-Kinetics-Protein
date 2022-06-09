#################################################
# Author: Dr. Fernando Arce Vega                #
# Email: farce@cio.mx                           #
# Date: 02/06/2022                              #
# Subject: Computing binding constants - GUI    #
#################################################


# Libraries
import sys
import time
import os

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from PyQt5.uic import loadUi
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Graph_indicator(FigureCanvas):
    def __init__(self):

        # Figure settings
        self.fig = Figure(figsize = (200, 100), dpi = 100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        FigureCanvas.__init__(self, self.fig)
        self.fig.canvas.draw()

    def add_data(self, Concentrations, t_mnts, original, predicted, froot):
        """
        Description: this function plots the intensity of the measurements in dots and the predictions in continuous lines.
        Inputs: Concentrations and t_mnts are numpy-arrays and original and predicted lists.
        Returns: nothing.
        """

        # Graph settings
        self.ax.clear()
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.line_width = 1.2

        self.x = t_mnts
        self.y = np.zeros(len(self.x))

        # To determine the top margin in Y
        max_Y_v = []
        cnt = 0

        # Plotting graphs
        for i in range(0, len(original)*2, 2):
            col = self.colors[cnt]
            lab = 'C' + str(int(i/2) + 1)+ '= ' + str(Concentrations[int(i/2)])

            exec("self.plot_" + str(i + 1) + ", = self.ax.plot(self.x, self.y, c = col, lw = self.line_width, linestyle = 'dotted', label = lab)")
            exec("self.plot_" + str(i + 2) + ", = self.ax.plot(self.x, self.y, c = col, lw = self.line_width)")

            R = original[int(i/2)]
            P = predicted[int(i/2)]

            eval('self.plot_' + str(i + 1) + '.set_data(t_mnts, R)')
            eval('self.plot_' + str(i + 2) + '.set_data(t_mnts, P)')

            R_vect = np.concatenate([R, P])
            max_Y_v.append(np.max(R_vect))

            cnt += 1

            if cnt == len(self.colors):
                cnt = 0

        # More figure settings
        self.ax.set_ylim([0, np.max(max_Y_v) + 0.05])
        self.ax.set_xlim([-2, np.max(t_mnts) + 2])
        self.ax.set_xlabel('Time (minutes)')
        self.ax.set_ylabel('Signal intensity (A.U.)')
        self.ax.set_title('Data adjustment')
        self.ax.legend(loc = 'best')
        self.fig.canvas.draw()

        # Saving image
        fname = os.path.split(froot)[1]
        fname = fname[0:fname.rfind('.')]
        froot = os.path.split(froot)[0]

        try: # froot + os.sep + 'output' + os.sep + fname + '.png'
            self.fig.savefig('output' + os.sep + fname + '.png', dpi = 150)

        except:
            print('The image cannot be saved, check write permissions!')


class GUIapp(QMainWindow):
    def __init__(self):
        # Loading GUI        
        super(GUIapp, self).__init__()
        loadUi('gui.ui', self)
        
        # Disabling and connecting buttons
        self.calculate_pushButton.setEnabled(False)
        self.load_pushButton.clicked.connect(self.capture_info)
        self.calculate_pushButton.clicked.connect(self.compute_constants)
        
        # Placing graph in Widget
        self.graph = Graph_indicator()
        self.graph_Layout.addWidget(self.graph)

        # Displaying Widgets
        self.show()


    def capture_info(self):
        """
        Description: this function loads the measurements.
        Inputs: nothing.
        Returns: nothing.
        """

        # compatible files
        filter = 'CSV files (*.csv *.xlsx)'
        dialog = QFileDialog()
        self.froot = dialog.getOpenFileName(None, 'Window name', '', filter)[0]

        # File name
        self.fname = os.path.split(self.froot)[1]
        self.fname = self.fname[0:self.fname.rfind('.')]

        # Reading data
        if self.froot != '':
            self.textEdit.setPlainText('File:')
            self.textEdit.append(self.froot)
            self.textEdit.append('')

            if self.froot.endswith('.xlsx'):
              self.data = pd.read_excel(self.froot)

            elif self.froot.endswith('.csv'):
              self.data = pd.read_csv(self.froot)

            else:
                self.textEdit.append('Your file is not supported!')

            # Data description
            try:
                data_label_html = self.data.describe().to_html(border = 0)
                data_label_string = self.data.describe().to_string()           
                self.textEdit.append('')
                self.textEdit.append('Data statistics:')
                self.textEdit.append(data_label_html)

                # Saving data statistics
                try:
                    with open('output' + os.sep + self.fname + '.txt', 'w') as self.file_data:
                        self.file_data.write('File:\n')
                        self.file_data.write(self.froot)                        
                        self.file_data.write('\n\n')
                        self.file_data.write('Data statistics:\n')
                        self.file_data.write(data_label_string)
                        self.file_data.write('\n\n')
                        self.file_data.close()

                except:
                    self.textEdit.append('The ouput-file cannot be generated, check write permissions!')

                # Enabling and disabling buttons
                self.calculate_pushButton.setEnabled(True)
                self.load_pushButton.setEnabled(False)

            except:
                self.textEdit.append("Your file's format is not supported!")

                # Enabling and disabling buttons
                self.calculate_pushButton.setEnabled(False)
                self.load_pushButton.setEnabled(True)


    def compute_constants(self):
        """
        Description: this function compute the binding constants.
        Inputs: nothing.
        Returns: nothing.
        """

        # Enabling and disabling buttons
        self.load_pushButton.setEnabled(False)
        self.textEdit.append('')

        def relative_error(arr):
            """
            Description: this function compute the relative error of the measruments.
            Inputs: An array.
            Returns: A floating number.
            """

            arr_n = len(arr)
            arr_mean = np.mean(arr)
            arr_error_abs = (np.sum((arr - arr_mean)**2) / (arr_n*(arr_n - 1)))**(1/2)

            return arr_error_abs / arr_mean

        try:
            # Required time
            self.tic = time.time()

            # Extracting data
            # Concentrations
            C_M = self.data[-2:].to_numpy()
            C_M = C_M[0]

            C_mL = self.data[-1:].to_numpy()
            C_mL = C_mL[0]

            # Measurements
            data = self.data[:-2]

            # Measurements time
            time_data = data.iloc[:, 0].to_numpy().astype(np.float32)
            measurements = data.iloc[:, 1::].to_numpy()

            # Concatenating time 0
            t_mnts = time_data / 60
            t_mnts = np.concatenate([np.array([0]), t_mnts])

            # Concatenating measurements at time 0
            patterns_n, measurements_n = np.shape(measurements)

            zeros = np.zeros((1, measurements_n))
            Rtp = np.concatenate([zeros, measurements])


            def function(t_mnts, Plateau, K, Y0):
                """
                Description: this function approximates the parameters: Plateau, K, Y0 to the model Y=Y0 + (Plateau-Y0)*(1-exp(-K*x)) using Lavender-Marquardt algorithm.
                Inputs: t_mnts and R are numpy-array.
                Returns: Plateau, K and Y0 and float numbers.
                """

                return Y0 + (Plateau - Y0) * (1 - np.exp(-K*t_mnts))

            # Concatenating parameters
            Y0_v = ['Y0']
            Plateau_v = ['Plateau']
            K_v = ['K(s^-1)']
            R2_v = ['R^2']
            Span_v = ['Span']
            R_v = []
            predictions_v = []

            # For each measurements
            for i in range(measurements_n):
                R = Rtp[:, i]

                # Approximating parameters using the Lavender-Marquardt algorithm     
                popt, _ = curve_fit(function, t_mnts, R)
                Plateau, K, Y0 = popt
                span = Plateau - abs(Y0)

                Plateau_v.append(Plateau)
                K_v.append(K)
                Y0_v.append(Y0)
                Span_v.append(span)

                # Prediction of the current measurement
                prediction = Y0 + (Plateau - Y0) * (1 - np.exp(-K*t_mnts[1::]))
                R_v.append(R[1::])
                predictions_v.append(prediction)

                # R2 of the prediction
                R2 = r2_score(prediction, R[1::])
                R2_v.append(R2)

            # Plot data
            self.graph.add_data(C_M[1::], t_mnts[1::], R_v, predictions_v, self.froot)

            # Computing Rmax
            Rmax = np.max(Span_v[1::])

            # One phase association table
            C_M_s = C_M.copy()
            C_mL_s = C_mL.copy()
            Y0_v_s = Y0_v.copy()
            Plateau_v_s = Plateau_v.copy()
            K_v_s = K_v.copy()
            R2_v_s = R2_v.copy()
            Span_v_s = Span_v.copy()

            for i in range(measurements_n):
              C_mL_s[i + 1] = '{:.3f}'.format(C_mL[i + 1])
              C_M_s[i + 1] = '{:.3f}'.format(C_M[i + 1]*1e9)
              Y0_v_s[i + 1] = '{:.3f}'.format(Y0_v[i + 1]) 
              Plateau_v_s[i + 1] = '{:.3f}'.format(Plateau_v[i + 1]) 
              K_v_s[i + 1] = '{:.3f}'.format(K_v[i + 1])
              Span_v_s[i + 1] = '{:.3f}'.format(Span_v[i + 1])
              R2_v_s[i + 1] = '{:.3f}'.format(R2_v[i + 1])

            Phase_table = PrettyTable()
            Phase_table.border = False
            Phase_table.field_names = C_mL_s 
            Phase_table.add_row(C_M_s)
            Phase_table.add_row(Y0_v_s)
            Phase_table.add_row(Plateau_v_s)
            Phase_table.add_row(K_v_s)
            Phase_table.add_row(Span_v_s)
            Phase_table.add_row(R2_v_s)

            self.textEdit.append('')
            Phase_table_html = Phase_table.get_html_string()
            Phase_table_string = Phase_table.get_string()
            self.textEdit.append('One phase association parameters:')
            self.textEdit.append(Phase_table_html)
            self.textEdit.append('')
            self.textEdit.append('')

            # Saving computed parameters
            try:
                with open('output' + os.sep + self.fname + '.txt', 'a') as self.file_data:
                    self.file_data.write('One phase association parameters:\n')
                    self.file_data.write(Phase_table_string)
                    self.file_data.write('\n\n')
                    self.file_data.close()

            except:
                self.textEdit.append('The ouput-file cannot be generated, check write permissions!')

            # Computing Ka, Kd, and KD
            Ka_avg_v_s = ['Ka ']
            Ka_error_rel_v_s = []
            Kd_avg_v_s = ['Kd ']
            Kd_error_rel_v_s = []
            KD_avg_v_s = ['KD ']
            KD_error_rel_v_s = []
            header = ['ng m𝐋^-1', 'Ka (M^-1 S^-1)', 'Kd (S^-1)', 'KD (M)']

            data_table = PrettyTable()
            data_table.border = False
            data_table.field_names = header

            for i in range(measurements_n - 1):
                R = Rtp[1:, i + 1]
                K = K_v[i + 2]
                C = C_M[i + 2]

                # Computing binding constants
                Ka = (K * R) / (C * Rmax * (1 - np.exp(-K*time_data)))
                Kd = K - C*Ka
                KD = Kd/Ka

                # Ka relative error
                Ka_avg = np.mean(Ka)
                Ka_error_rel = relative_error(Ka)
                Ka_error_rel_v_s.append(Ka_error_rel)

                # Kd relative error
                Kd_avg = np.mean(Kd)
                Kd_error_rel = relative_error(Kd)
                Kd_error_rel_v_s.append(Kd_error_rel)

                # KD relative error
                KD_avg = np.mean(KD)
                KD_error_rel = relative_error(KD)
                KD_error_rel_v_s.append(KD_error_rel)

                # Data concatenation
                lab = C_mL[i + 2]
                Ka_string = str(format(Ka_avg,'.3E')) + ' (±' + str(format(Ka_error_rel_v_s[i],'.3E')) + ')  '
                Kd_string = str(format(Kd_avg,'.3E')) + ' (±' + str(format(Kd_error_rel_v_s[i],'.3E')) + ')  '
                KD_string = str(format(KD_avg,'.3E')) + ' (±' + str(format(KD_error_rel_v_s[i],'.3E')) + ')  '

                row = [lab, Ka_string, Kd_string, KD_string]

                Ka_avg_v_s.append(Ka_avg)
                Kd_avg_v_s.append(Kd_avg)
                KD_avg_v_s.append(KD_avg)
                data_table.add_row(row)

            # Computing Ka average relative error
            Ka_avg_error_rel = relative_error(np.array(Ka_avg_v_s[1::], dtype = np.float32))

            # Computing Kd average relative error
            Kd_avg_error_rel = relative_error(np.array(Kd_avg_v_s[1::], dtype = np.float32))

            # Computing KD average relative error
            KD_avg_error_rel = relative_error(np.array(KD_avg_v_s[1::], dtype = np.float32))

            # Printing table
            Ka_avg_string = format(np.mean(Ka_avg_v_s[1::]), '.3E') + ' (±' + str(format(Ka_avg_error_rel, '.3E')) + ')  '
            Kd_avg_string = format(np.mean(Kd_avg_v_s[1::]), '.3E') + ' (±' + str(format(Kd_avg_error_rel, '.3E')) + ')  '
            KD_avg_string = format(np.mean(KD_avg_v_s[1::]), '.3E') + ' (±' + str(format(KD_avg_error_rel, '.3E')) + ')  '

            data_table.add_row(['Average ', Ka_avg_string, Kd_avg_string, KD_avg_string])

            # Required time
            self.toc = time.time()
            self.d_time = self.toc - self.tic

            data_table_html = data_table.get_html_string()
            data_table_string = data_table.get_string()
            self.textEdit.append('Binding constants:')
            self.textEdit.append(data_table_html)
            self.textEdit.append('')
            self.textEdit.append('')
            self.textEdit.append('Required time:')
            self.textEdit.append('{:.3f} seconds'.format(self.d_time))

            # Saving computed constants
            try:
                with open('output' + os.sep + self.fname + '.txt', 'a') as self.file_data:
                    self.file_data.write('Binding constants:\n')
                    self.file_data.write(data_table_string)
                    self.file_data.write('\n\n')
                    self.file_data.write('Required time:\n')
                    self.file_data.write('{:.3f} seconds'.format(self.d_time))
                    self.file_data.close()

            except:
                self.textEdit.append('The ouput-file cannot be generated, check write permissions!')

            # Enabling and disabling buttons
            self.load_pushButton.setEnabled(True)
            self.calculate_pushButton.setEnabled(False)

        except:
            self.textEdit.append("Your file's format is not supported!")

            try:
                with open('output' + os.sep + self.fname + '.txt', 'a') as self.file_data:
                    self.file_data.write("Your file's format is not supported!")
                    self.file_data.close()

            except:
                self.textEdit.append('The ouput-file cannot be generated, check write permissions!')

            # Enabling and disabling buttons
            self.calculate_pushButton.setEnabled(False)
            self.load_pushButton.setEnabled(True)


# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    appWindow = GUIapp()
    status = app.exec_()
    sys.exit(status)