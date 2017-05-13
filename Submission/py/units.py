#===============================================================================
""" Import Modules """
#-------------------------------------------------------------------------------

import numpy as np
import pdb

#===============================================================================
""" Known Constants """
#-------------------------------------------------------------------------------

# dictionary of constants and parameters in SI units
SI          =   {}

# dictionary of constants and parameters that are the same between SI and model units
const       =   {}

const['A']  =   6.02214199e23       # Avagadro's number             (#)
SI['m_H']   =   1.6737236e-27       # mass of hydrogen              (kg)
SI['m_He']  =   6.6464764e-27       # mass of helium                (kg)
SI['mol_H'] =   1.00794/1000        # mol mass of hydrogen          (kg/mol)
SI['mol_He']=   4.002602/1000       # mol mass of helium            (kg/mol)
SI['d_H']   =   53e-12              # size of hydrogen atom         (m)
SI['d_He']  =   31e-12              # size of helium atom           (m)
SI['c']     =   2.99792458e8        # speed of light                (m/s)
SI['G']     =   6.67408e-11         # gravitational constant        (N m^2/kg^2)
SI['k']     =   1.3806503e-23       # Bolztmann's constant          (J/K)
SI['sigma'] =   5.670373e-8         # Stefan Bolztmann's constant   W/(m^2 K^4)
SI['a']     =   7.5657e-16          # radiation constant            J/(m^3 K^4)
SI['R']     =   8.314472            # gas constant                  J/(mol K)
SI['r_sun'] =   6.95508e8           # solar radius                  m

""" 3/4 hydrogen 1/4 helium cloud """
SI['mu']    =   (3/4)*SI['m_H'] + (1/4)*SI['m_He']      # average particle mass     (kg)
SI['mol_mu']=   (3/4)*SI['mol_H'] + (1/4)*SI['mol_He']  # average particle mol mass (kg/mol)
SI['d']     =   (3/4)*SI['d_H'] + (1/4)*SI['d_He']      # average particle size     (m)

#===============================================================================
""" Model Parameters """
#-------------------------------------------------------------------------------

# const['gamma']  =   5/3     # heat capacity ratio for monatomic ideal gas
const['gamma']  =   1.41    # heat capacity ratio for diatomic gas (H and He)
const['T']      =   10      # initial cloud temperature (K)
const['T_c']    =   10**7   # critical temperature for H -> He fusion

#===============================================================================
""" Conversion factors """
#-------------------------------------------------------------------------------

names           =   {'length':'pc',
                     'time':'Myr',
                     'mass':'M$_\odot$',
                     'temp':'K'
                     }

# names           =   {'length':'m',
#                      'time':'s',
#                      'mass':'kg',
#                      'temp':'K'
#                      }

# unit_length     =   1
# unit_time       =   1
# unit_mass       =   1

unit_length     =   1/3.086e16                  # m -> pc
unit_time       =   1/(60*60*24*365.25*1e6)     # s -> Myr
unit_mass       =   1/1.9891e30                 # kg -> solar mass
unit_speed      =   unit_length / unit_time
unit_acc        =   unit_length / unit_time**2
unit_force      =   unit_mass * unit_length / unit_time**2
unit_energy     =   unit_force * unit_length
unit_power      =   unit_energy / unit_time
unit_pressure   =   unit_force / unit_length**2
unit_density    =   unit_mass / unit_length**3

solar_lum       =   1/3.828e26                  # W -> solar luminosity
solar_mass      =   1/1.9891e30                 # kg -> solar mass
Myr             =   1/(60*60*24*365.25*1e6)     # s -> Myr
pc              =   1/3.086e16                  # m -> pc

#===============================================================================
""" Convert to model units """
#-------------------------------------------------------------------------------

def unit_conversion():
    """ converts SI units to model units and returns dictionary of values"""
    # dictionary of constants and parameters in model units
    MD = {}

    MD['m_H']   =   SI['m_H'] * unit_mass
    MD['m_He']  =   SI['m_He'] * unit_mass
    MD['mol_H'] =   SI['mol_H'] * unit_mass
    MD['mol_He']=   SI['mol_He'] * unit_mass
    MD['d_H']   =   SI['d_H'] * unit_length
    MD['d_He']  =   SI['d_He'] * unit_length
    MD['c']     =   SI['c'] * unit_speed
    MD['G']     =   SI['G'] * unit_force * unit_length**2 / unit_mass**2
    MD['k']     =   SI['k'] * unit_energy
    MD['sigma'] =   SI['sigma'] * unit_power / unit_length**2
    MD['a']     =   SI['a'] * unit_energy / unit_length**3
    MD['R']     =   SI['R'] * unit_energy
    MD['mu']    =   SI['mu'] * unit_mass
    MD['mol_mu']=   SI['mol_mu'] * unit_mass
    MD['d']     =   SI['d'] * unit_length
    MD['r_sun'] =   SI['r_sun'] * unit_length
    return MD
""" dictionary of constants and parameters
in model units: pc, Myr, solar mass"""
MD              =   unit_conversion()

#===============================================================================
""" Final Dictionary of Constants """
#-------------------------------------------------------------------------------

C               =   {**const , **MD, **names}
# C               =   {**const, **SI, **names}
