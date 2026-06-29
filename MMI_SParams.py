import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

#CONSTANTS
    #a note on units c=1 and a=1um so all meep units are in microns (including time)
SIN_epsilon = 1.974
SIO2_epsilon = 1.4657

#device parameters
sm_waveguide_length=50
sm_waveguide_width=1
sm_waveguide_spacing=8
mm_waveguide_length=360
mm_waveguide_width=24
taper_length=50
taper_output_width=5

def GenerateTaperGeometry(taper_length, taper_input_width, taper_output_width, start_position_x, start_position_y, direction):
    if direction == 'right':
        taper_verticies = [mp.Vector3(start_position_x, start_position_y - taper_input_width/2),
                                    mp.Vector3(start_position_x + taper_length, start_position_y - taper_output_width/2),
                                    mp.Vector3(start_position_x + taper_length, start_position_y + taper_output_width/2),
                                    mp.Vector3(start_position_x, start_position_y + taper_input_width/2)]
    elif direction == 'left':
        taper_verticies = [mp.Vector3(start_position_x, start_position_y - taper_input_width/2),
                                    mp.Vector3(start_position_x - taper_length, start_position_y - taper_output_width/2),
                                    mp.Vector3(start_position_x - taper_length, start_position_y + taper_output_width/2),
                                    mp.Vector3(start_position_x, start_position_y + taper_input_width/2)]
    else:
        raise ValueError("Direction must be 'right' or 'left'")
    taper = mp.Prism(taper_verticies,height=mp.inf,material=mp.Medium(epsilon=SIN_epsilon))
    return taper

def GenerateWaveguideGeometry(waveguide_length, waveguide_width, start_position_x, start_position_y):
    waveguide_verticies = [mp.Vector3(start_position_x, start_position_y - 0.5*waveguide_width),
                                         mp.Vector3(start_position_x + waveguide_length, start_position_y - 0.5*waveguide_width),
                                         mp.Vector3(start_position_x + waveguide_length, start_position_y + 0.5*waveguide_width),
                                         mp.Vector3(start_position_x, start_position_y + 0.5*waveguide_width)]
    waveguide = mp.Prism(waveguide_verticies,height=mp.inf,material=mp.Medium(epsilon=SIN_epsilon))
    return waveguide

def GenerateMMIGeometry(   sm_waveguide_length, sm_waveguide_width,
                                                sm_waveguide_spacing,
                                                mm_waveguide_length, mm_waveguide_width, 
                                                taper_length, taper_output_width):
    """Generates the geometry for a 2x2 MMI coupler with the specified parameters."""
    cell_x_um = mm_waveguide_length + 2*sm_waveguide_length + 2*taper_length
    cell_y_um = np.ceil(1.2 * mm_waveguide_width)

    P1_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, -cell_x_um/2 + sm_waveguide_length, sm_waveguide_spacing/2, 'right')
    P2_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, -cell_x_um/2 + sm_waveguide_length, -sm_waveguide_spacing/2, 'right')
    P3_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, cell_x_um/2 - sm_waveguide_length, sm_waveguide_spacing/2, 'left')
    P4_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, cell_x_um/2 - sm_waveguide_length, -sm_waveguide_spacing/2, 'left')

    P1_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, -cell_x_um/2, sm_waveguide_spacing/2)
    P2_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, -cell_x_um/2, -sm_waveguide_spacing/2)
    P3_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, cell_x_um/2 - sm_waveguide_length, sm_waveguide_spacing/2)
    P4_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, cell_x_um/2 - sm_waveguide_length, -sm_waveguide_spacing/2)
    
    mm_waveguide = GenerateWaveguideGeometry(mm_waveguide_length, mm_waveguide_width, -cell_x_um/2 + sm_waveguide_length + taper_length, 0)
    cell = mp.Vector3(cell_x_um,cell_y_um,0)
    geometry = [P1_taper, P2_taper, P3_taper, P4_taper, P1_waveguide, P2_waveguide, P3_waveguide, P4_waveguide, mm_waveguide]
    # geometry = [P1_waveguide, P2_waveguide, P3_waveguide, P4_waveguide, mm_waveguide]

    return geometry, cell


def RunMMISimulation(wavelength_um, resolution):
    geometry, cell = GenerateMMIGeometry(   sm_waveguide_length=sm_waveguide_length,
                                                    sm_waveguide_width=sm_waveguide_width,
                                                    sm_waveguide_spacing=sm_waveguide_spacing,
                                                    mm_waveguide_length=mm_waveguide_length,
                                                    mm_waveguide_width=mm_waveguide_width,
                                                    taper_length=taper_length,
                                                    taper_output_width=taper_output_width)
    cell_x_um = cell[0]
    cell_y_um = cell[1]
    frequency = 1/wavelength_um
    sources = [
    mp.GaussianBeamSource(
        src=mp.ContinuousSource(frequency),
        center=mp.Vector3(-cell_x_um/2 + 1.5, -sm_waveguide_spacing/2),
        size=mp.Vector3(0,5*sm_waveguide_width),
        beam_kdir=mp.Vector3(1,0),
        beam_w0=0.8*sm_waveguide_width,
        beam_E0=mp.Vector3(0,0,1),
        )
    ]
    pml_layers = [mp.PML(1.0)]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        default_material=mp.Medium(epsilon=SIO2_epsilon))

    sim.run(until=SIN_epsilon*cell_x_um) #light has just reached the end of the device
    return sim

def VisualiseDevice(sim, LOG=True):
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Ez)
    fig = plt.figure(figsize=(100,30))
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    if LOG:
        plt.imshow(np.log(np.abs(ez_data.transpose())+1), alpha=0.9, )
    else:
        plt.imshow(ez_data.transpose(), cmap="RdBu", alpha=0.9, )

    plt.xticks((np.arange(0, sim.cell_size[0] * sim.resolution, 100)))
    plt.savefig("MMI_Device.png", dpi=300)


def VisualiseTransverseFields(sim):
    pass

def VisualiseLongitudinalFields(sim):
    pass


def GetSParameters(sim, waveguide_spacing):
    resolution = sim.resolution
    ez_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Ez)

    #longitudinal field profiles
    P3_ypos = ez_data.shape[1]//2 - waveguide_spacing//2 * resolution
    P4_ypos = ez_data.shape[1]//2 + waveguide_spacing//2 * resolution
    P1_ypos = ez_data.shape[1]//2 - waveguide_spacing//2 * resolution
    P3_Efield = ez_data[-13*resolution:-3 *resolution , P3_ypos]
    P4_Efield = ez_data[-13*resolution:-3*resolution, P4_ypos]
    P1_Efield = ez_data[3*resolution: 13*resolution, P1_ypos]

    window = np.kaiser(len(P3_Efield),6)
    edge_size =     int(np.ceil(1.5*resolution))
    # window = np.ones_like(output1)
    hP1 = sp.hilbert(P1_Efield*window)
    hP3 = sp.hilbert(P3_Efield*window)
    hP4 = sp.hilbert(P4_Efield*window)

    def HilbertSParam(hx, hy, edge):
        Hxy = hx/hy
        Sxy = np.mean(Hxy[edge:-edge])
        return Sxy
    
    S31 = HilbertSParam(hP3, hP1, edge_size)
    S41 = HilbertSParam(hP4, hP1, edge_size)
    return S31, S41

def PrintSParameters(Sxy, name="Sxy"):
    amplitude_dB = 20*np.log10(np.abs(Sxy))
    phase_deg = np.degrees(np.angle(Sxy))
    print(f"{name}: {Sxy:.4f}, Amplitude: {amplitude_dB:.2f} dB, ∠: {phase_deg:.2f} degrees")
    
def main(wavelength_um=1.55, resolution=12, visualise=False):
    #global import meep
    global mp
    import meep as mp
    sim = RunMMISimulation(wavelength_um, resolution)
    if visualise:
        VisualiseDevice(sim)
        print("Wavelength: {:.4f} nm".format(wavelength_um * 1000))
        PrintSParameters(S31, name="S31")
        PrintSParameters(S41, name="S41")
    S31, S41 = GetSParameters(sim, sm_waveguide_spacing)
    return np.array([[S31, S41], [S41, S31]])


def MMI_SMatrix(wavelength_um, example=False, precomputed=False):
    if example:
        return np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
    if precomputed:
        # Return precomputed S-parameters if available
        pass
    return main(wavelength_um=wavelength_um, resolution=12, visualise=False)


if __name__ == "__main__":
    wavelength_um = 1.55 - 6*0.0001761
    resolution = 12
    main(wavelength_um=wavelength_um, resolution=resolution, visualise=True)