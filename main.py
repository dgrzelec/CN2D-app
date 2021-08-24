from graph import *
import itertools

from kivy.clock import Clock
from kivy.core.window import WindowBase
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.utils import get_color_from_hex as rgb
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListLayout
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.app import App
import numpy as np
from settings import settings_json_system, settings_json_wavepocket, settings_json_animation, settings_json_potential, settings_json_magfield
from kivy.uix.settings import SettingsWithTabbedPanel

from scipy.sparse.linalg import dsolve
import scipy.sparse as sparse
from numba import jit
import pickle
import os
import threading
import time


@jit
def evolveCN2D(PsiTab, Vtab, SMatrix, B, f, dd, L, W ):
#psitab size is LxW

    for i in range(L):
        for j in range(W):
            if j != 0 and j != W-1 and i != L-1:
                B[i * W + j] = -0.5 * f * ( PsiTab[i-1][j] + PsiTab[i+1, j] + PsiTab[i][j-1] + PsiTab[i][j+1] ) + (2*f + 0.5*Vtab[i][j] + dd) * PsiTab[i][j]
            elif j == 0 and i != L-1:
                B[i * W + j] = -0.5 * f * (PsiTab[i - 1][j] + PsiTab[i + 1, j] + PsiTab[i][j + 1]) + (2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
            elif j == W-1 and i != L-1:
                B[i * W + j] = -0.5 * f * (PsiTab[i - 1][j] + PsiTab[i + 1, j] + PsiTab[i][j - 1]) + (2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
            elif i == L-1:
                if j != 0 and j != W-1:
                    B[i * W + j] = -0.5 * f * (PsiTab[i - 1][j] + PsiTab[0, j] + PsiTab[i][j - 1] + PsiTab[i][j + 1]) + (2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
                elif j == 0:
                    B[i * W + j] = -0.5 * f * (PsiTab[i - 1][j] + PsiTab[0, j] + PsiTab[i][j + 1]) + (2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
                elif j == W-1:
                    B[i * W + j] = -0.5 * f * (PsiTab[i - 1][j] + PsiTab[0, j] + PsiTab[i][j - 1]) + (2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
    #print("Vector created..",end="  ")


    print("Vector created..")
    psi_new = dsolve.spsolve(SMatrix, B, use_umfpack=True)

    #     #with mumps - test  #######################################################
    # import scipy.sparse.linalg as sla
    # import kwant.linalg.mumps
    #
    # class LuInv(sparse.linalg.LinearOperator):
    #     def __init__(self, A):
    #         inst = kwant.linalg.mumps.MUMPSContext()
    #         inst.factor(A, ordering='metis')
    #         self.solve = inst.solve
    #         sla.LinearOperator.__init__(self, A.dtype, A.shape)
    #
    #     def _matvec(self, x):
    #         return self.solve(x.astype(self.dtype))
    #
    # solver = LuInv(SMatrix)
    # psi_new = solver(B)

    return psi_new


def evolveCN2D_with_B(PsiTab, Vtab, SMatrix, B, f, dd, L, W, phi):
    # psitab size is LxW

    for i in range(L):
        for j in range(W):
            if j != 0 and j != W - 1 and i != L - 1:
                B[i * W + j] = -0.5 * f * (
                            PeiersPhase(j, phi, 1) * PsiTab[i - 1][j] + PeiersPhase(j, phi, -1) * PsiTab[i + 1, j] + PsiTab[i][j - 1] + PsiTab[i][j + 1])\
                            + (2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
            elif j == 0 and i != L - 1:
                B[i * W + j] = -0.5 * f * (PeiersPhase(j, phi, 1)*PsiTab[i - 1][j] + PeiersPhase(j, phi, -1)*PsiTab[i + 1, j] + PsiTab[i][j + 1]) + (
                            2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
            elif j == W - 1 and i != L - 1:
                B[i * W + j] = -0.5 * f * (PeiersPhase(j, phi, 1)*PsiTab[i - 1][j] + PeiersPhase(j, phi, -1)*PsiTab[i + 1, j] + PsiTab[i][j - 1]) + (
                            2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
            elif i == L - 1:
                if j != 0 and j != W - 1:
                    B[i * W + j] = -0.5 * f * (
                                PeiersPhase(j, phi, 1)*PsiTab[i - 1][j] + PeiersPhase(j, phi, -1)*PsiTab[0, j] + PsiTab[i][j - 1] + PsiTab[i][j + 1]) + (
                                               2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
                elif j == 0:
                    B[i * W + j] = -0.5 * f * (PeiersPhase(j, phi, 1)*PsiTab[i - 1][j] + PeiersPhase(j, phi, -1)*PsiTab[0, j] + PsiTab[i][j + 1]) + (
                                2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
                elif j == W - 1:
                    B[i * W + j] = -0.5 * f * (PeiersPhase(j, phi, 1)*PsiTab[i - 1][j] + PeiersPhase(j, phi, -1)*PsiTab[0, j] + PsiTab[i][j - 1]) + (
                                2 * f + 0.5 * Vtab[i][j] + dd) * PsiTab[i][j]
    print("Vector B created..", end="  ")
    psi_new = dsolve.spsolve(SMatrix, B, use_umfpack=True)
    return psi_new

class WavePocket2D:
    def __init__(self, sigma_x, sigma_y, k):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.k = k

    def set_k(self, kk):
        self.k = kk

    def value(self, center, system_size):
        x_0, y_0 = center
        g2D_tab = np.zeros((system_size[1], system_size[0]), dtype="complex")
        X = np.arange(0, system_size[0]) #rosnący X ( system_size[1] == L )
        for y in range(system_size[1]): #system_size[1] = W
            g2D_tab[y] =  1/(2*np.pi*self.sigma_x*self.sigma_y) * np.exp(-( (X - x_0)**2/(2*self.sigma_x**2) + (y - y_0)**2/(2*self.sigma_y**2) ) + 1j*self.k*X) + 0.j

        return g2D_tab

    def value_2(self, center, system_size, canal_width, n):
        x_0, y_0 = center
        g2D_tab = np.zeros((system_size[0], system_size[1]), dtype="complex")
        #         X = np.arange(0, system_size[0]) #rosnący X ( system_size[1] == L )
        #         for y in range(system_size[1]): #system_size[1] = W
        #             if y < y_0 - int(canal_width/2) or y > y_0 + int(canal_width/2):
        #                 g2D_tab[y] = np.zeros((system_size[0]))
        #             else:
        #                 g2D_tab[y] =  1/(2*np.pi*self.sigma_x) * np.exp(-( (X - x_0)**2/(2*self.sigma_x**2) ) + 1j*self.k*X) * np.sqrt(2/canal_width)*np.sin(n*np.pi/canal_width*(y-y_0+int(canal_width/2))) + 0.j

        for i in range(system_size[0]):
            for j in range(system_size[1]):
                if j < y_0 - int(canal_width / 2) or j > y_0 + int(canal_width / 2):
                    g2D_tab[i, j] = 0
                else:
                    g2D_tab[i, j] = 1 / (2 * np.pi * self.sigma_x) * np.exp(
                        -((i - x_0) ** 2 / (2 * self.sigma_x ** 2)) + 1j * self.k * i) * np.sqrt(
                        2 / canal_width) * np.sin(n * np.pi / canal_width * (j - y_0 + int(canal_width / 2))) + 0.j
        g2D_tab = g2D_tab.transpose()
        return g2D_tab

class PotentialShape():
    def __init__(self, L, W, V, cw=40):
        self.v_value = V
        self.L =L
        self.W = W
        self.center_x = int(self.L/2)
        self.center_y = int(self.W/2)
        self.width = 5
        self.a = 5
        self.canal_width = cw

    def single_barrier(self):
        Vtab = np.zeros((self.L, self.W), dtype="complex")

        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(self.W):
                Vtab[i,j] = self.v_value

        return Vtab
    def single_slit(self):
        Vtab = self.single_barrier()

        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(int(self.W/2) - self.a, int(self.W/2) + self.a + 1):
                Vtab[i,j] = 0

        return Vtab

    def double_slit_wide(self):

        Vtab = self.single_barrier()

        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(int(self.W/2) - 2*self.a - 8, int(self.W/2) + 2*self.a + 8 + 1):
                if j < int(self.W/2) - self.a or j > int(self.W/2) + self.a:
                    Vtab[i,j] = 0

        return Vtab

    def double_slit_narrow(self):
        Vtab = self.single_barrier()

        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(int(self.W/2) - 2*self.a - 5, int(self.W/2) + 2*self.a + 5 + 1):
                if j < int(self.W/2) - self.a or j > int(self.W/2) + self.a:
                    Vtab[i,j] = 0

        return Vtab

    def obstacle(self):
        Vtab = self.single_barrier()

        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(int(self.W/2) - int(self.W/5), int(self.W/2) + int(self.W/5) + 1):
                if j < int(self.W/2) - self.a -2 or j > int(self.W/2) + self.a + 2:
                    Vtab[i,j] = 0

        return Vtab

    def double_slit_close(self):
        Vtab = self.single_barrier()

        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(int(self.W/2) - 2*self.a - 2, int(self.W/2) + 2*self.a + 2 + 1):
                if j < int(self.W/2) - 2 or j > int(self.W/2) + 2:
                    Vtab[i,j] = 0

        return Vtab

    def corner(self):
        Vtab = self.single_barrier()
        for i in range(self.center_x - self.width, self.center_x + self.width + 1):
            for j in range(0, int(self.W/2) + 1):
                if j < int(self.W/2) - 2 or j > int(self.W/2) + 2:
                    Vtab[i,j] = 0
        return Vtab

    def two_terminal_ring(self):
        Vtab = np.ones((self.L, self.W), dtype="complex")
        Vtab = self.v_value * Vtab

        self.o_radius = 120
        self.i_radius = 80
        self.width = self.canal_width

        for i in range(self.center_x - self.o_radius, self.center_x + self.o_radius + 1):
            for j in range(self.center_y - self.o_radius, self.center_y + self.o_radius):
                if np.sqrt((i-self.center_x)**2 + (j-self.center_y)**2) > self.i_radius and np.sqrt((i-self.center_x)**2 + (j-self.center_y)**2) < self.o_radius:
                    Vtab[i, j] = 0

        for i in range(0, self.L):
            for j in range(self.center_y - int(self.width/2), self.center_y + int(self.width/2)):
                if i < self.center_x - self.i_radius or i > self.center_x + self.i_radius:
                    Vtab[i,j] = 0

        return Vtab

    def two_terminal_ring_smooth(self, r, offset, offset2, cut):
        Vtab = np.ones((self.L, self.W))
        Vtab = self.v_value * Vtab

        self.o_radius = 120
        self.i_radius = 80
        self.width = self.canal_width

        for i in range(0, self.L):
            for j in range(self.center_y - int(self.width / 2), self.center_y + int(self.width / 2)):
                if i < self.center_x - self.i_radius or i > self.center_x + self.i_radius:
                    Vtab[i, j] = 0

        #         offset = 0
        #         r = 40

        for i in range(self.center_x - self.o_radius - r + offset, self.center_x - self.o_radius + offset):
            for j in range(self.center_y + int(self.canal_width / 2) - offset2,
                           self.center_y + int(self.canal_width / 2) + r - offset2 - cut):
                if np.sqrt((i - (self.center_x - self.o_radius - r + offset)) ** 2 + (
                        j - (self.center_y + int(self.canal_width / 2) + r - offset2)) ** 2) > r:
                    Vtab[i, j] = 0

        for i in range(self.center_x + self.o_radius - offset, self.center_x + self.o_radius + r - offset):
            for j in range(self.center_y + int(self.canal_width / 2) - offset2,
                           self.center_y + int(self.canal_width / 2) + r - offset2 - cut):
                if np.sqrt((i - (self.center_x + self.o_radius + r - offset)) ** 2 + (
                        j - (self.center_y + int(self.canal_width / 2) + r - offset2)) ** 2) > r:
                    Vtab[i, j] = 0

        for i in range(self.center_x + self.o_radius - offset, self.center_x + self.o_radius + r - offset):
            for j in range(self.center_y - int(self.canal_width / 2) - r + offset2 + cut,
                           self.center_y - int(self.canal_width / 2) + offset2):
                if np.sqrt((i - (self.center_x + self.o_radius + r - offset)) ** 2 + (
                        j - (self.center_y - int(self.canal_width / 2) - r + offset2)) ** 2) > r:
                    Vtab[i, j] = 0

        for i in range(self.center_x - self.o_radius - r + offset, self.center_x - self.o_radius + offset):
            for j in range(self.center_y - int(self.canal_width / 2) - r + offset2 + cut,
                           self.center_y - int(self.canal_width / 2) + offset2):
                if np.sqrt((i - (self.center_x - self.o_radius - r + offset)) ** 2 + (
                        j - (self.center_y - int(self.canal_width / 2) - r + offset2)) ** 2) > r:
                    Vtab[i, j] = 0

        for i in range(self.center_x - self.o_radius, self.center_x + self.o_radius + 1):
            for j in range(self.center_y - self.o_radius, self.center_y + self.o_radius):
                if np.sqrt((i - self.center_x) ** 2 + (j - self.center_y) ** 2) > self.i_radius and np.sqrt(
                        (i - self.center_x) ** 2 + (j - self.center_y) ** 2) < self.o_radius:
                    Vtab[i, j] = 0
        return Vtab

    def canal(self):

        Vtab = np.ones((self.L, self.W), dtype="complex")
        Vtab = self.v_value * Vtab

        width = self.canal_width

        for i in range(0, self.L):
            for j in range(self.center_y - int(width/2), self.center_y + int(width/2)):
                    Vtab[i,j] = 0

        return Vtab

class ImaginaryPotential(PotentialShape):

    def Manolopoulos(self, c, delta, x1, x2, f):

        a = 1 - 16/c**3
        b = (1 - 17/c**3)/c**2

        Vtab = np.zeros((self.L, self.W), dtype="complex")

        Emin = f*(c/(2*delta*(x2-x1)))**2
        k_min = np.sqrt( 1/f*Emin )

        for i in range(x1+1, x2):
            x = 2*k_min*delta/(i-x1)
            for j in range(self.W):

                Vtab[i,j] = -1j*Emin*( a*x - b*x**3 + 4/(c - x)**2 - 4/(c+x)**2 )

        return Vtab

def PeiersPhase(j, phi, sign=-1):

    return np.exp(1j*sign*phi*2*j)


def save_data_to_file(data_list, filename='last_data'):

    with open('data/'+filename + '.data', 'wb') as file:
        pickle.dump(data_list, file)

def read_data(path='data/last_data.data'):

    with open(path, 'rb') as file:
        temp = pickle.load(file)

    return temp

class CN2DApp(App):

    def build_A_matrix(self):
        #self.update_bar_trigger = Clock.create_trigger(self.update_progress_bar)
        # rozwiązanie blokowe
        # tworze potrzebne bloki
        self.B_0 = sparse.lil_matrix((self.W, self.W), dtype="complex")
        self.B_1 = self.B_0.copy()
        self.B_1.setdiag(0.5 * self.f)
        self.B_2 = self.B_0.copy()
        for i in range(1, self.W - 1):
            self.B_2[i, i + 1] = 0.5 * self.f
            self.B_2[i, i - 1] = 0.5 * self.f

        self.B_2[0, 1] = 0.5 * self.f
        self.B_2[self.W - 1, self.W - 2] = 0.5 * self.f
        #
        # self.A_list = []
        # for i in range(int(self.L/100)):
        #     self.A_list.append(sparse.lil_matrix((self.W, self.W*self.L), dtype="complex"))


        # z tych bloków tworzymy macierz A
        # najpierw tworze rzędy: pierwszy, drugi, przedostatni, ostatni. środkowe wygodnie mozna już pętlą
        self.zeros = sparse.hstack([self.B_0] * (self.L - 3))
        self.row1 = sparse.hstack([self.B_2, self.B_1, self.zeros, self.B_1])
        self.row2 = sparse.hstack([self.B_1, self.B_2, self.B_1, self.zeros])
        self.rowL_1 = sparse.hstack([self.zeros, self.B_1, self.B_2, self.B_1])
        self.rowL = sparse.hstack([self.B_1, self.zeros, self.B_1, self.B_2])

        self.A = sparse.vstack([self.row1, self.row2])

        for i in range(1, self.L - 3):
            if i % 10 == 0:
                print("Row {}".format(i))
                self.calc_pb.value = i/(self.L-3)*100

                self.p_label.text = "Contructing A matrix: {:.1%}".format(self.calc_pb.value/100)

            self.temprow = sparse.hstack([sparse.hstack([self.B_0] * i), self.B_1, self.B_2, self.B_1,
                                          sparse.hstack([self.B_0] * (self.L - 3 - i))])
            self.A = sparse.vstack([self.A, self.temprow])

###########################################################################################
        #
        # threads = list()
        # for index in range(int(self.L/100)):
        #     mm = (index+1)*100 + 1
        #     x = threading.Thread(target=self.thread_part_A_mat_construction, args=(index, mm if mm < self.L - 3 else self.L - 3, ))
        #     threads.append(x)
        #     x.start()
        #
        # for index, thread in enumerate(threads):
        #     thread.join()
        #     self.A = sparse.vstack([self.A, self.A_list[index]])
##############################################################################################
        self.A = sparse.vstack([self.A, self.rowL_1, self.rowL])

        self.A.setdiag(self.d_vector)
        # print(self.A.toarray())
        self.A = sparse.csc_matrix(self.A)

    def thread_part_A_mat_construction(self, index, max_row):
        temp_mat = list()
        for i in range(index*100+1, max_row):
            if i % 10 == 0 and index == 0:
                print("Row {}".format(i))
                self.calc_pb.value = i / max_row * 100
                self.p_label.text = "Contructing A matrix: {:.1%}".format(self.calc_pb.value / 100)

            temprow = sparse.hstack([sparse.hstack([self.B_0] * i), self.B_1, self.B_2, self.B_1,
                                          sparse.hstack([self.B_0] * (self.L- 3 - i))])
            if i > 1:
                temp_mat = sparse.vstack([temp_mat, temprow])
            else:
                temp_mat = temprow
        #self.A_list[index] = sparse.lil_matrix((self.W*(max_row - index*100-1), self.L*self.W), dtype="complex")
        self.A_list[index] = temp_mat

    def build_A_matrix_with_B(self):

        self.B_0 = sparse.lil_matrix((self.W, self.W), dtype="complex")
        self.B_diag = self.B_0.copy()

        for i in range(1, self.W - 1):
            self.B_diag[i, i + 1] = 0.5 * self.f
            self.B_diag[i, i - 1] = 0.5 * self.f
        self.B_diag[0, 1] = 0.5 * self.f
        self.B_diag[self.W - 1, self.W - 2] = 0.5 * self.f


        self.B_minus = self.B_0.copy()
        for j in range(0,self.W):
            self.B_minus[j,j] = 0.5 * self.f * PeiersPhase(j, self.phi, 1)


        self.B_plus = self.B_0.copy()
        for j in range(0,self.W):
            self.B_plus[j,j] = 0.5 * self.f * PeiersPhase(j, self.phi, -1)

        # z tych bloków tworzymy macierz A
        # najpierw tworze rzędy: pierwszy, drugi, przedostatni, ostatni. środkowe wygodnie mozna już pętlą
        self.zeros = sparse.hstack([self.B_0] * (self.L - 3))
        self.row1 = sparse.hstack([self.B_diag, self.B_plus, self.zeros, self.B_minus])
        self.row2 = sparse.hstack([self.B_minus, self.B_diag, self.B_plus, self.zeros])
        self.rowL_1 = sparse.hstack([self.zeros, self.B_minus, self.B_diag, self.B_plus])
        self.rowL = sparse.hstack([self.B_plus, self.zeros, self.B_minus, self.B_diag])

        self.A = sparse.vstack([self.row1, self.row2])

        for i in range(1, self.L - 3):
            if i % 10 == 0:
                print("Row {}".format(i))
                self.calc_pb.value = i / (self.L - 3) * 100

                self.p_label.text = "Contructing A matrix: {:.1%}".format(self.calc_pb.value / 100)

            self.temprow = sparse.hstack([sparse.hstack([self.B_0] * i), self.B_minus, self.B_diag, self.B_plus,
                                          sparse.hstack([self.B_0] * (self.L - 3 - i))])
            self.A = sparse.vstack([self.A, self.temprow])

        self.A = sparse.vstack([self.A, self.rowL_1, self.rowL])

        self.A.setdiag(self.d_vector)
        # print(self.A.toarray())
        self.A = sparse.csc_matrix(self.A)


    def build_config(self, config):
        config.setdefaults('System',{
            'dt': 45,
            'dx': 1,
            'L': 200,
            'W': 100,
            'calc_ene': False
        })
        config.setdefaults('WavePocket',{
            'k': 0.015,
            'sigma_x': 10,
            'sigma_y': 20,
            'center_x': 20,
            'center_y': 50,
            'transverse': True,
            'n': 1,
            'canal_width': 40,
        })
        config.setdefaults('Animation',{
            'freq': 5,
            'max_ticks': 20,
            'graph_size': 800
        })
        config.setdefaults('Potential', {
            'v_value': 8,
            'shape': "single_barrier",
            'imaginary': False,
            'c': 2.62206,
            'delta': 0.2,
            'x1': 720,
            'x2': 1040
        })
        config.setdefaults('MagneticField', {
            'state': True,
            'phi': 0.2
        })
    def build_settings(self, settings):
        settings.add_json_panel('System',
                                self.config, data=settings_json_system)
        settings.add_json_panel('WavePocket',
                                self.config, data=settings_json_wavepocket)
        settings.add_json_panel('Animation',
                                self.config, data=settings_json_animation)
        settings.add_json_panel('Potential',
                                self.config, data=settings_json_potential)
        settings.add_json_panel('MagneticField',
                                self.config, data=settings_json_magfield)

    def on_config_change(self, config, section, key, value):
        self.set_settings()
        if section == "System" or section=="Potential" or section=="MagneticField": self.system_changed = True


    def set_settings(self):
        #system panel
        self.dx =int(self.config.get('System', 'dx'))
        self.dt = int(self.config.get('System', 'dt'))
        self.m = 1
        self.h_bar = 1
        self.L = int(self.config.get('System', 'L'))  # lengh of the system - in 0.1 nm
        self.W = int(self.config.get('System', 'W'))  # width
        self.calc_ene = False if self.config.get('System', 'calc_ene') is '0' else True

        self.f = self.h_bar / (2*self.m*self.dx*self.dx)
        self.dd = 1j * self.h_bar / self.dt
        #wavepocket panel
        self.k = float(self.config.get('WavePocket', 'k'))
        self.sigma_x = int(self.config.get('WavePocket', 'sigma_x'))
        self.sigma_y = int(self.config.get('WavePocket', 'sigma_y'))
        self.centex_x = int(self.config.get('WavePocket', 'center_x'))
        self.centex_y = int(self.config.get('WavePocket', 'center_y'))
        self.transverse = False if self.config.get('WavePocket', 'transverse') is '0' else True
        self.n = int(self.config.get('WavePocket', 'n'))
        self.canal_width = int(self.config.get('WavePocket', 'canal_width'))
        #animation panel
        self.freq = int(self.config.get('Animation', 'freq'))
        self.rate = 1.0 / self.freq
        self.max_ticks = int(self.config.get('Animation', 'max_ticks'))
        self.graph_size = int(self.config.get('Animation', 'graph_size'))
        #potetnial panel
        if self.config.get("Potential","shape") == "single_barrier":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value"))*self.f).single_barrier()
        elif self.config.get("Potential","shape") == "single_slit":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value"))*self.f).single_slit()
        elif self.config.get("Potential","shape") == "double_slit_wide":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value"))*self.f).double_slit_wide()
        elif self.config.get("Potential","shape") == "double_slit_narrow":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value"))*self.f).double_slit_narrow()
        elif self.config.get("Potential", "shape") == "obstacle":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f).obstacle()
        elif self.config.get("Potential", "shape") == "double_slit_close":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f).double_slit_close()
        elif self.config.get("Potential", "shape") == "corner":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f).corner()
        elif self.config.get("Potential", "shape") == "two_terminal_ring":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f, self.canal_width ).two_terminal_ring()
        elif self.config.get("Potential", "shape") == "two_terminal_ring_smooth":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f, self.canal_width ).two_terminal_ring_smooth(70,24,0, 19)
        elif self.config.get("Potential", "shape") == "canal":
            self.Vtab = PotentialShape(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f,self.canal_width).canal()
        else: self.Vtab = np.zeros((self.L, self.W), dtype="complex")

        self.imaginary = False if self.config.get("Potential", "imaginary") is '0' else True
        self.c = float(self.config.get("Potential", "c"))
        self.delta = float( self.config.get("Potential", "delta") )
        self.x1 = int( self.config.get("Potential", "x1") )
        self.x2 = int( self.config.get("Potential", "x2") )

        #magfield panel
        self.phi = float(self.config.get('MagneticField', 'phi'))
        self.magnetic_field_state = False if self.config.get('MagneticField', 'state') is '0' else True
        print(self.magnetic_field_state, type(self.magnetic_field_state)) #debug

    def update_progress_bar(self, value):
        self.calc_pb.value = value


    def build(self):
        self.use_kivy_settings = False
        self.settings_cls = SettingsWithTabbedPanel
        config = self.config
        WindowBase.set_title(self, "Cranc_Nicolson in 2D aplication")

        print(self.config.get('System','dx'))
        #print(os.listdir('data/'))
        #domyślne ustawienia, zmiana ich również wyżej
        # self.dx = 1
        # self.dt = 45
        # self.m = 1
        # self.h_bar = 1
        # self.L = 200    #lengh of the system - in 0.1 nm
        # self.W = 100    #width
        #
        # self.k = 0.015
        # self.sigma_x = 10
        # self.sigma_y = 20
        # self.centex_x = 20
        # self.centex_y = 50
        #
        # self.freq = 5
        #self.rate = int(1 / self.freq)
        # self.max_ticks = 20

        self.set_settings()
        self.system_changed = True

        self.system_size = (self.L, self.W)
        #self.Vtab = np.zeros(self.system_size)
        #self.Vtab = PotentialShape(self.L, self.W, 20*self.f).single_barrier()


        if self.transverse:
            self.PsiMatrix = WavePocket2D(self.sigma_x, self.sigma_y, self.k).value_2( ( self.centex_x, self.centex_y) ,self.system_size, self.canal_width, self.n )
        else:
            self.PsiMatrix = WavePocket2D(self.sigma_x, self.sigma_y, self.k).value((self.centex_x, self.centex_y),
                                                                                      self.system_size)

        #print(self.PsiMatrix.shape)
        self.PsiMatrix = self.PsiMatrix.transpose()
        print(self.PsiMatrix.shape)
        print((self.config.get('System', 'dt')))

        self.f = self.h_bar / (2*self.m*self.dx**2)
        self.dd = 1j * self.h_bar / self.dt

        self.PsiVector = np.resize(self.PsiMatrix, (self.W*self.L,))

        if self.imaginary:
            self.Vtab += ImaginaryPotential(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f).Manolopoulos(self.c, self.delta, self.x1, self.x2, self.f)

        self.VVector = np.resize(self.Vtab, (self.W*self.L,))
        self.d_vector = np.zeros(self.L*self.W, dtype="complex")

        for i in range(self.L*self.W):
            self.d_vector[i] = -2*self.f - 0.5*self.VVector[i] + self.dd

        self.calculations_done = False

        self.count = 0
        #print(type(self.config.get('System', 'dx')))

        self.create_graph()

        self.Energy_list = []

################################################# BUTTONS AND WIDGETS #############################################################3


        self.calc_btn = Button(size_hint=(1.0, 0.5) ,text='Calculate')

        self.calc_pb = ProgressBar(max=100.0, size_hint=(1.0, 0.2)) #calculations progress bar
        self.p_label = Label(text="[b]Waiting for user[/b]", markup=True ,size_hint=(1.0, 0.2))

        self.sim_btn = Button(size_hint = (1.0, 0.5), text = "Start animation")
        self.stop_btn = Button(size_hint = (1.0, 0.5), text = "Stop animation")
        self.redraw_btn = Button(size_hint = (1.0, 0.5), text = "Redraw system") 
        self.generate_btn = Button(size_hint = (1.0, 0.5), text = "Generate") ######################################### zbindować

        self.fname_texbox = TextInput(text="", multiline=False)
        self.load_btn = Button(size_hint = (1.0, 0.5), text = "Load data")
        self.save_btn = Button(size_hint=(1.0, 0.5), text="Save data")

        self.calc_btn.bind(on_press=self.calc_btn_callback)
        self.sim_btn.bind(on_press = self.animation_btn_callback)
        self.stop_btn.bind(on_press=self.stop_btn_callback)
        self.redraw_btn.bind(on_press = self.redraw_btn_callback)
        self.generate_btn.bind(on_press = self.generate_btn_callback)

        self.save_btn.bind(on_press=self.save_btn_callback)
        self.load_btn.bind(on_release=self.load_btn_callback1) #dropdown creation

##################################### LAYOUTS ###################################################

        self.box_vertical = BoxLayout(padding=10, orientation='vertical')
        self.box_horizontal = BoxLayout(padding=5, orientation='horizontal')
        self.box_vertical_middle = BoxLayout(padding=5,spacing=10, orientation='vertical')
        self.box_vertical_left = BoxLayout(padding=5,spacing=10, orientation='vertical')
        self.box_vertical_right = BoxLayout(padding=5,spacing=10, orientation='vertical')

        #self.box_vertical.add_widget(self.graph2)
        self.box_vertical.add_widget(self.graph)


        self.box_vertical_middle.add_widget(self.calc_pb)
        self.box_vertical_middle.add_widget(self.p_label)
        self.box_vertical_left.add_widget(self.redraw_btn)
        self.box_vertical_left.add_widget(self.calc_btn)
        self.box_vertical_left.add_widget(self.generate_btn)
        self.box_vertical_right.add_widget(self.save_btn)
        self.box_vertical_right.add_widget(self.fname_texbox)
        self.box_horizontal.add_widget(self.box_vertical_left)
        self.box_horizontal.add_widget(self.box_vertical_middle)
        self.box_horizontal.add_widget(self.box_vertical_right)
        self.box_horizontal.add_widget(self.load_btn)
        self.box_horizontal.size_hint_max_y = 250
        self.box_horizontal.size_hint_min_y = 150
        self.box_vertical_left.size_hint_max_x = 400
        self.box_vertical_middle.size_hint_max_x = 400
        self.box_vertical_right.size_hint_max_x = 400
        self.box_vertical.add_widget(self.box_horizontal)


        #Clock.schedule_interval(self.update_contour, 1 / 60.)

        return self.box_vertical
################################################ graph creation ###########################################
    def create_graph(self):

        graph_theme = {
            'label_options': {
                'color': rgb('444444'),  # color of tick labels and titles
                'bold': True},
            'background_color': rgb('f8f8f2'),  # back ground color of canvas
            'tick_color': [0,0,0,1],  # ticks and grid
            'border_color': [0,0,0,1]}  # border drawn around each graph


        self.graph = Graph(
            xlabel='x',
            ylabel='y',
            #x_ticks_minor=10,
            x_ticks_major=50,
            y_ticks_major=50,
            y_grid_label=True,
            x_grid_label=True,
            padding=10,
            xlog=False,
            ylog=False,
            x_grid=False,
            y_grid=False,
            xmin=0,
            xmax=self.L if self.L >= self.W else self.W,
            ymin=0,
            ymax=self.W if self.W >= self.L else self.L,
            **graph_theme)

        self.data_list = []
        plot = ContourPlot()
        temp = self.PsiMatrix.copy()
        temp = temp.transpose()
        temp = np.abs(temp*temp.conj())

        if self.L > self.W and (self.L-self.W)%2 == 0:
            t = int((self.L - self.W)/2)
            z = np.zeros((t, self.L))
            temp = np.vstack((z, temp, z))

            plot.xrange = (0, self.L)
            plot.yrange = (0, self.L)
        elif self.L<self.W and (self.W-self.L)%2 == 0:
            t = (self.W - self.L)/2
            z = np.zeros((temp, self.L))
            temp = np.concatenate((z, temp, z))

            plot.xrange = (0, self.W)
            plot.yrange = (0, self.W)
        else:
            plot.xrange = (0, self.L)
            plot.yrange = (0, self.W)


        plot.data = temp
        # plot.xrange = (0, self.L)
        # plot.yrange = (0, self.W)
        plot.color = [1, 1, 1, 1]
    ################################### second graph, ontop first, for displaying barriers (potencial) // abadoned
        # self.graph2 = Graph(
        #     xlabel='x',
        #     ylabel='y',
        #     # x_ticks_minor=10,
        #     x_ticks_major=50,
        #     y_ticks_major=50,
        #     y_grid_label=True,
        #     x_grid_label=True,
        #     padding=10,
        #     xlog=False,
        #     ylog=False,
        #     x_grid=False,
        #     y_grid=False,
        #     xmin=0,
        #     xmax=self.L,
        #     ymin=0,
        #     ymax=self.W,
        #     **graph_theme)
        #
        # plot2 = ContourPlot()
        # temp2 = self.Vtab.copy()
        # temp2 = temp2.transpose()
        # plot2.data = temp2/self.config.getint("Potential", "v_value")*self.PsiMatrix.max().real
        # plot2.xrange = (0, self.L)
        # plot2.yrange = (0, self.W)
        # plot2.color = [.4, .4, .4, 0.5]

        self.contourplot = plot
        #plot.color = [46/255, 139/255, 87/255, 1]
        self.graph.add_plot(plot)
        # self.graph2.add_plot(plot2)



        self.graph.pos_hint = {'center_x': 0.5, 'center_y': 1}
        self.graph.size_hint = (None, None)
        self.graph.size = (self.graph_size, self.graph_size) ##############################################################################################################zmienic

        # if self.L > self.W:
        #     self.graph.size = (2*self.L, 2*self.L)
        # elif self.L < self.W:
        #     self.graph.size = (2 * self.W, 2 * self.W)
        # else:
        #     self.graph.size = (2 * self.L, 2 * self.W)


        # self.graph2.pos_hint = {'center_x': 0.5, 'center_y': 1}
        # self.graph2.size_hint = (None, None)
        # self.graph2.size = (self.graph_size * self.L / self.W, self.graph_size)
        #self.graph2.pos = self.graph.pos

    ########################################### CALBACKS ###################################################################
    def calc_btn_callback(self, event):
        if self.calculations_done:
            self.box_vertical_left.remove_widget(self.sim_btn)
            self.box_vertical_left.remove_widget(self.stop_btn)
            self.calculations_done = False
        #starting calculating simulation
        threading.Thread(target=self.threaded_calc).start()

    def threaded_calc(self):
        self.t_max = self.dt * self.max_ticks
        t=0
        self.p_label.text = "Constructuing A matrix..."
        if self.system_changed == True:

            if self.magnetic_field_state == False:
                self.build_A_matrix()
            else:
                self.build_A_matrix_with_B()

            self.system_changed = False

        self.PsiVector = self.PsiVector / np.linalg.norm(self.PsiVector)  # norming initial state

        self.B = np.zeros((self.L * self.W), dtype="complex")
        self.p_label.text = "Starting simulation calculations..."

        T0 = time.perf_counter()

        while t<=self.t_max:
            self.PsiMatrix = np.resize(self.PsiVector, (self.L, self.W))


            if self.magnetic_field_state == False:
                self.Psi_new = evolveCN2D(self.PsiMatrix, self.Vtab, self.A, self.B, self.f, self.dd, self.L, self.W)
                evolveCN2D.inspect_types()
            else:
                self.Psi_new = evolveCN2D_with_B(self.PsiMatrix, self.Vtab, self.A, self.B, self.f, self.dd, self.L, self.W, self.phi)

            self.p_label.text = "Solved {:.1%} ".format(t/self.t_max)
            self.calc_pb.value = t/self.t_max*100
            print("Solved {}".format(int(t/self.dt)))

            self.data_list.append( self.Psi_new )

            self.PsiVector = self.Psi_new

            t += self.dt

            if int(t/self.dt)%10 == 0:
                self.make_contour_data(int(t/self.dt -1))
                self.contourplot.data[:] = self.plot_data
                self.contourplot.ask_draw()



        self.calculations_done = True
        self.p_label.text = "Calculations done!"
        self.calc_pb.value = 100

        print( "Time elapsed: {}".format(time.perf_counter() - T0) ) #to benchmark calculations time

        self.p_label.text = "Saving..."
        save_data_to_file(self.data_list)

        if self.calc_ene:
            threading.Thread(target=self.energy_calc).start()

        self.p_label.text = "Done!"

        self.box_vertical_left.add_widget(self.sim_btn)
        self.box_vertical_left.add_widget(self.stop_btn)


    def animation_btn_callback(self, event):
        self.count1 = 0
        Clock.schedule_interval(self.update_contour, self.rate)

    def stop_btn_callback(self, event):
        if self.count1 > 0:
            Clock.unschedule(self.update_contour)

    def redraw_btn_callback(self, event):
        self.set_settings()

        self.data_list = []
        self.Energy_list = []
        self.system_size = (self.L, self.W)

        if self.transverse:
            self.PsiMatrix = WavePocket2D(self.sigma_x, self.sigma_y, self.k).value_2((self.centex_x, self.centex_y),
                                                                                      self.system_size,
                                                                                      self.canal_width, self.n)
        else:
            self.PsiMatrix = WavePocket2D(self.sigma_x, self.sigma_y, self.k).value((self.centex_x, self.centex_y),
                                                                                    self.system_size)
        # print(self.PsiMatrix.shape)
        self.PsiMatrix = self.PsiMatrix.transpose()
        print(self.PsiMatrix.shape)
        print(self.config.items('WavePocket'))
        print(self.k)
        self.PsiVector = np.resize(self.PsiMatrix, (self.W * self.L,))

        if self.imaginary:
            self.Vtab += ImaginaryPotential(self.L, self.W, int(self.config.get("Potential", "v_value")) * self.f).Manolopoulos(self.c, self.delta, self.x1, self.x2, self.f)

        self.VVector = np.resize(self.Vtab, (self.W*self.L,))
        self.d_vector = np.zeros(self.L*self.W, dtype="complex")

        for i in range(self.L*self.W):
            self.d_vector[i] = -2*self.f - 0.5*self.VVector[i] + self.dd


        self.box_vertical.remove_widget(self.graph)
        self.box_vertical_left.remove_widget(self.sim_btn)

        self.create_graph()

        self.box_vertical_left.remove_widget(self.sim_btn)
        self.box_vertical_left.remove_widget(self.stop_btn)
        self.box_vertical.clear_widgets()


        self.box_vertical.add_widget(self.graph)
        self.box_vertical.add_widget(self.box_horizontal)

        self.calc_pb.value = 0


        self.p_label.text = "System ready"


    def save_btn_callback(self, event):

        system_settings = {'dt': self.config.get('System', 'dt'),
                           'dx': self.config.get('System', 'dx'),
                            'L': self.config.get('System', 'L'),
                            'W': self.config.get('System', 'W')}
        wave_pocket_settings =  {'k': self.config.get('WavePocket', 'k'),
                                 'sigma_x': self.config.get('WavePocket', 'sigma_x'),
                                 'sigma_y': self.config.get('WavePocket', 'sigma_y'),
                                 'center_x': self.config.get('WavePocket', 'center_x'),
                                 'center_y': self.config.get('WavePocket', 'center_y')}
        anim_settings = {'max_ticks': self.config.get('Animation', 'max_ticks')}

        save_data_to_file((self.data_list, system_settings, wave_pocket_settings, anim_settings), self.fname_texbox.text)

    def load_btn_callback1(self, event):

        self.load_dropdown = DropDown()

        for fname in os.listdir('data/'):
            btn = Button(text=fname, size_hint_y=None, height=20)

            btn.bind(on_release=lambda btn: self.load_dropdown.select(btn.text))
            self.load_dropdown.add_widget(btn)
        self.load_dropdown.open(self.load_btn)
        self.load_dropdown.bind(on_select = self.load_btn_callback2)

    def load_btn_callback2(self, event, fname):
        if fname == "last_data.data":
            self.redraw_btn_callback(self.redraw_btn)
            self.data_list = read_data('data/'+fname)

            self.box_vertical_left.add_widget(self.sim_btn)
            self.box_vertical_left.add_widget(self.stop_btn)
        else:
            d_list, sys, waves, anim = read_data('data/'+fname)
            self.config.setall("System", sys)
            #print("center_y= {}".format(waves.get("center_y")))
            if waves.get("center_y") is None:
                waves['center_y'] = int(self.W/2)
            self.config.setall("WavePocket", waves)
            self.config.setall("Animation", anim)
            self.config.write()

            self.redraw_btn_callback(self.redraw_btn)
            self.data_list = d_list

            self.box_vertical_left.add_widget(self.sim_btn)
            self.box_vertical_left.add_widget(self.stop_btn)

        if self.calc_ene:
            threading.Thread(target=self.energy_calc).start()

    def generate_btn_callback(self, event):
        threading.Thread(target=self.generate).start()

    def generate(self):

        self.p_label.text = "Generating..."


        self.PsiVector = self.PsiVector / np.linalg.norm(self.PsiVector)  # norming initial state

        self.B = np.zeros((self.L * self.W), dtype="complex")

        def save_data_to_file_g(data_list, filename='last_data'):

            with open('generated/' + filename + '.data', 'wb') as file:
                pickle.dump(data_list, file)


        system_settings = {'dt': self.config.get('System', 'dt'),
                           'dx': self.config.get('System', 'dx'),
                           'L': self.config.get('System', 'L'),
                           'W': self.config.get('System', 'W')}
        wave_pocket_settings = {'k': self.config.get('WavePocket', 'k'),
                                'sigma_x': self.config.get('WavePocket', 'sigma_x'),
                                'sigma_y': self.config.get('WavePocket', 'sigma_y'),
                                'center_x': self.config.get('WavePocket', 'center_x'),
                                'center_y': self.config.get('WavePocket', 'center_y')}
        anim_settings = {'max_ticks': self.config.get('Animation', 'max_ticks')}

        save_data_to_file_g((self.d_vector, self.Vtab, self.B, self.PsiVector, self.f, self.dd, self.magnetic_field_state, self.phi, system_settings, wave_pocket_settings, anim_settings),
                          self.fname_texbox.text)

        self.p_label.text = ''
        self.calc_pb.value = 0

#######################################  NORM AND ENERGY   #######################################################################################
    def norm(self, index=0):
        suma = self.data_list[index].copy()
        suma = (np.abs(suma)**2)*(self.dx**2)
        suma = np.sum(suma)

        return suma

    def energy(self, index=0):
        E_list = []
        E=0
        vector = self.data_list[index]
        for i in range(self.L):
            for j in range(self.W):
                if j != 0 and j != self.W - 1 and i != self.L - 1:
                    E_list.append(np.conj(vector[i * self.W + j])*( -self.f*vector[(i-1) * self.W + j]
                                                                -self.f*vector[(i+1) * self.W + j]
                                                                -self.f*vector[(i) * self.W + j-1]
                                                                -self.f*vector[(i) * self.W + j +1]
                                                                +(4*self.f + self.Vtab[i][j])*vector[(i) * self.W + j] ) * self.dx**2
                                  )
                elif j == 0 and i != self.L - 1:
                    E_list.append( np.conj(vector[i * self.W + j]) * (
                                -self.f * vector[(i - 1) * self.W + j]
                                - self.f * vector[(i + 1) * self.W + j]
                                - self.f * vector[(i) * self.W + j + 1]
                                + (4 * self.f + self.Vtab[i][j]) * vector[
                                    (i) * self.W + j]) * self.dx ** 2
                                   )
                elif j == self.W - 1 and i != self.L - 1:
                    E_list.append(np.conj(vector[i * self.W + j]) * (
                                -self.f * vector[(i - 1) * self.W + j]
                                - self.f * vector[(i + 1) * self.W + j]
                                - self.f * vector[(i) * self.W + j - 1]
                                + (4 * self.f + self.Vtab[i][j]) * vector[
                                    (i) * self.W + j]) * self.dx ** 2
                                  )
                elif i == self.L - 1:
                    if j != 0 and j != self.W - 1:
                        E_list.append( np.conj(vector[i * self.W + j]) * (
                                    -self.f * vector[(i - 1) * self.W + j]
                                    - self.f * vector[0 * self.W + j]
                                    - self.f * vector[(i) * self.W + j - 1]
                                    - self.f * vector[(i) * self.W + j + 1]
                                    + (4 * self.f + self.Vtab[i][j]) * vector[
                                        (i) * self.W + j]) * self.dx ** 2
                                       )

                    elif j == 0:

                        E_list.append( np.conj(vector[i * self.W + j]) * (
                                    -self.f * vector[(i - 1) * self.W + j]
                                    - self.f * vector[0 * self.W + j]
                                    - self.f * vector[(i) * self.W + j + 1]
                                    + (4 * self.f + self.Vtab[i][j]) * vector[
                                        (i) * self.W + j]) * self.dx ** 2
                                       )
                    elif j == self.W - 1:
                        E_list.append( np.conj(vector[i * self.W + j]) * (
                                    -self.f * vector[(i - 1) * self.W + j]
                                    - self.f * vector[0 * self.W + j]
                                    - self.f * vector[(i) * self.W + j - 1]
                                    + (4 * self.f + self.Vtab[i][j]) * vector[
                                        (i) * self.W + j]) * self.dx ** 2
                                       )
        return np.abs(np.sum(E_list))

    def energy_calc(self):
        self.p_label.text = "Calculating energy..."
        for index in range(len(self.data_list)):
            self.Energy_list.append( self.energy(index) )
            self.calc_pb.value = index/self.max_ticks*100
            self.p_label.text = "Calculating energy:  {:.1%}".format(index/self.max_ticks)

        self.p_label.text = ''
        self.calc_pb.value = 0


    


    ########################################## CONTOUR PLOT DATA PREPARATION FUNCTIONS ###################################################
    def make_contour_data(self, index=0):
        self.plot_data = self.data_list[index].copy()
        self.plot_data = np.resize(self.plot_data, (self.L, self.W))
        self.plot_data = self.plot_data.transpose()
        self.plot_data = np.abs( np.multiply(np.array(self.plot_data) ,np.array( self.plot_data.conj()) ) )

        if self.L > self.W and (self.L-self.W)%2 == 0:
            temp = int((self.L - self.W)/2)
            z = np.zeros((temp, self.L))
            self.plot_data = np.concatenate((z, self.plot_data, z))
        elif self.L<self.W and (self.W-self.L)%2 == 0:
            temp = (self.W - self.L)/2
            z = np.zeros((temp, self.L))
            self.plot_data = np.concatenate((z, self.plot_data, z))

    def update_contour(self, *args):

        self.make_contour_data(self.count1)
        self.contourplot.data[:] = self.plot_data

        self.contourplot.ask_draw()

        print("Drawing points with index {}, NORM: {} , ENERGY: {}".format(self.count1, self.norm(self.count1), self.Energy_list[self.count1] if self.Energy_list!= [] else "none")) #debuging
        #print("plot data shape: {}".format(np.shape(self.plot_data)))
        #print(self.plot_data[-1][-1])

        self.count1 += 1
        if self.count1 == self.max_ticks:
            Clock.unschedule(self.update_contour)

if __name__ == '__main__':
    CN2DApp().run()




