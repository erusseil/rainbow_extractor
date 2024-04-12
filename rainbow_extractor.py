import numpy as np
import pandas as pd
import fink_utils.photometry.conversion as convert
from light_curve.light_curve_py import RainbowFit
from light_curve.light_curve_py import bolometric as blm
from light_curve.light_curve_py import temperature as tp
import warnings
from light_curve.light_curve_py import warnings as rainbow_warnings
from collections import Counter
import matplotlib.pyplot as plt
from seaborn import color_palette
import os


warnings.filterwarnings("ignore", category=rainbow_warnings.ExperimentalWarning)

class Lightcurve():
    
    color_palette = color_palette("Set2")
    
    def __init__(self):
        self.objectid = None
        self.alertid = None
        self.flux = None
        self.fluxerr = None
        self.mjd = None
        self.band = None
        self.meta = None
        self.rising = -999
        self.biggest_band = None
        self.biggest_color = None
        self.non_detec = False
        self.true_class = None
        
    def time_shift(self):
        if len(self.mjd)>0:
            self.mjd -= self.mjd[np.argmax(self.flux)]
        
    def normalize(self):
        if len(self.flux)>0:
            factor = np.max(self.flux)
            self.flux /= factor
            self.fluxerr /= factor
            
    @staticmethod
    def group_days(x, rounded_mjd):
        return np.split(x, np.where(np.diff(rounded_mjd) != 0)[0]+1)

    @staticmethod
    def quad_err(x):
        return np.sqrt(np.sum(np.square(x)))/len(x)

    def intraday_average(self):
        new_mjd, new_flux, new_fluxerr, new_band = [], [], [], []

        for b in np.unique(self.band):

            submjd = self.mjd[self.band==b]
            subflux = self.flux[self.band==b]
            subfluxerr = self.fluxerr[self.band==b]

            rounded_mjd = np.trunc(submjd)

            grouped_mjd = self.group_days(submjd, rounded_mjd)
            grouped_flux = self.group_days(subflux, rounded_mjd)
            grouped_fluxerr = self.group_days(subfluxerr, rounded_mjd)

            add_mjd = [np.mean(i) for i in grouped_mjd]
            new_mjd += add_mjd
            new_flux += [np.mean(i) for i in grouped_flux]
            new_fluxerr += [self.quad_err(i) for i in grouped_fluxerr]
            new_band += [b] * len(add_mjd)

        mjd, flux, fluxerr, band = zip(*sorted(zip(new_mjd, new_flux, new_fluxerr, new_band)))

        self.mjd = np.array(mjd)
        self.flux = np.array(flux)
        self.fluxerr = np.array(fluxerr)
        self.band = np.array(band)
        
    def clean_nan(self):
        if len(self.flux)>0:
            # Find all nans accross arrays
            mask = (self.flux==self.flux) & (self.fluxerr==self.fluxerr) &\
            (self.band==self.band) & (self.mjd==self.mjd)

            self.mjd = self.mjd[mask]
            self.flux = self.flux[mask]
            self.fluxerr = self.fluxerr[mask]
            self.band = self.band[mask]
        
    def check_rising(self):
        
        if len(self.flux)>0:
            # Check if the lc is purely rising or falling:
            pure_rise, pure_fall = [], []
            for band in np.unique(self.band):
                subflux = self.flux[self.band==band]
                subfluxerr = self.fluxerr[self.band==band]

                if len(subflux)>1:
                    rise = all((subflux[:-1]-subfluxerr[:-1])<(subflux[-1]+subfluxerr[-1]))
                    fall = all((subflux[1:]-subfluxerr[1:])<(subflux[0]+subfluxerr[0]))
                else:
                    rise, fall = True, True
                
                pure_rise.append(rise)
                pure_fall.append(fall)
                
            # In the case where the light curve is flat, both pure_rise and pure fall could be true
            # In this case we arbitratly consider the lc to be rising.
            if all(pure_rise):
                self.rising = 1
            elif all(pure_fall):
                self.rising = -1
            else:
                self.rising = 0
        
    def plot(self, xlim=None, ylim=None, fit=None):
        if fit != None:
            if xlim == None:
                t_range = 0.05 * np.ptp(self.mjd)
                X_smooth = np.linspace(self.mjd.min() - t_range, self.mjd.max() + t_range, 200)
            else:
                X_smooth = np.linspace(xlim[0], xlim[1], 200)

        for idx, band in enumerate(np.unique(self.band)):
            flux = self.flux[self.band==band]
            fluxerr = self.fluxerr[self.band==band]
            mjd = self.mjd[self.band==band]
            plt.errorbar(mjd, flux, yerr=fluxerr, fmt='o', label=band, color=self.color_palette[idx])
            
            if fit != None:
                plt.plot(X_smooth, fit[0](X_smooth, band, *fit[1]), alpha=.7, linewidth=3, color=self.color_palette[idx])

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        if fit != None:
            plt.legend()
        plt.show()
        
    def count_points(self):
        if len(self.band) > 0:
            point_counts = dict(Counter(self.band))

            if self.non_detec != None:
                for band in self.non_detec:
                    point_counts[band] -= max(self.non_detec[band]-1, 0)

            _, ordered_point_counts = zip(*sorted(zip(point_counts.values(), point_counts)))

            self.biggest_band = point_counts[ordered_point_counts[-1]]
            if len(ordered_point_counts)>1:
                self.biggest_color = point_counts[ordered_point_counts[-2]]

        
    def full_format(self):
        self.clean_nan()
        self.normalize()
        self.time_shift() 
        self.intraday_average()
        self.check_rising()
    
class ZTF_lightcurve(Lightcurve):
    
    band_wave_aa = {1: 4770.0, 2: 6231.0, 3: 7625.0}
    
    
    def __init__(self, file):
        super().__init__()
        self.objectid, self.mjd, self.flux, self.fluxerr, self.band, self.non_detec, self.meta = self.read_csv(file)

    @staticmethod
    def read_csv(file):
        obj = pd.read_csv(file)
        objectid = obj['i:objectId'][0]
        
        obj['i:flux'], obj['i:fluxerr'] = convert.mag2fluxcal_snana(obj['i:magpsf'], obj['i:sigmapsf'])
        obj, non_detec = ZTF_lightcurve.upperlim_process(obj)
        
        flux, fluxerr = obj['i:flux'], obj['i:fluxerr']
        band = np.array(obj['i:fid'])
        mjd = obj['i:jd']
        mjd, flux, fluxerr, band = zip(*sorted(zip(mjd, flux, fluxerr, band)))
        mjd, flux, fluxerr, band = np.array(mjd), np.array(flux), np.array(fluxerr), np.array(band)
        
        additional = {'ra':obj['i:ra'][0],
                      'dec':obj['i:dec'][0]}
        
        return objectid, mjd, flux, fluxerr, band, non_detec, additional

    @staticmethod
    def upperlim_process(obj):
        
        '''In the future it should be changed so that we only use the last forced phot point'''
        
        if 'd:tag' in obj.keys():
            # Keep only upperlim before trigger
            trigger = obj['i:jd'][obj['d:tag'] != 'upperlim'].min()
            ok_non_detec = (obj['d:tag'] == 'upperlim') & (obj['i:jd'] < trigger)
            
            if any(ok_non_detec):

                non_detec = dict(Counter(obj[ok_non_detec]['i:fid']))
                obj = obj[ok_non_detec | (obj['d:tag'] != 'upperlim')]

                # Assign flux=0 and fluxerr = diffmaglim
                fluxlim, _ = convert.mag2fluxcal_snana(obj['i:diffmaglim'], [1]*len(obj['i:diffmaglim']))
                obj.loc[ok_non_detec, 'i:flux'] = 0
                obj.loc[ok_non_detec, 'i:fluxerr'] = 2 * fluxlim[ok_non_detec]
                obj.reset_index(inplace=True)

        return obj, non_detec
    
    
class ELAsTiCC_lightcurve(Lightcurve):
    
    band_wave_aa = {"u": 3751, "g": 4742, "r": 6173, "i": 7502, "z": 8679, "Y": 9711}
    keep_bands = ["g", "r", "i", "z"]
    
    # we don't use z_final_err because it is incorrect in ELASTICC
    object_metadata = ['hostgal_ellipticity', 'hostgal_mag_Y', 'hostgal_mag_g',\
                       'hostgal_mag_i', 'hostgal_mag_r', 'hostgal_mag_u',\
                       'hostgal_mag_z', 'hostgal_magerr_Y', 'hostgal_magerr_g',\
                       'hostgal_magerr_i', 'hostgal_magerr_r', 'hostgal_magerr_u',\
                       'hostgal_magerr_z', 'hostgal_snsep', 'hostgal_sqradius',\
                       'z_final', 'mwebv', 'mwebv_err']
    
    SN_like_dict = {'111':'Ia', '112':'Ib', '113':'II', '114':'Iax', '115':'91bg',\
                   '131':'SLSN', '132':'TDE', '135':'PISN'}
    
    def __init__(self, read_parquet_output):
        super().__init__()
        self.objectid, self.alertid, self.mjd, self.flux, self.fluxerr, self.band, self.non_detec, self.meta = read_parquet_output

    @staticmethod
    def read_parquet(file, n=None):
        data = pd.read_parquet(file)
        
        if n != None:
            data = data[:n]

        converted = data.apply(ELAsTiCC_lightcurve.convert_ELAsTiCC_format, axis=1)
        return [ELAsTiCC_lightcurve(i) for i in converted]
    
    @staticmethod
    def read_parquet_withclass(file, n=None):
        
        if 'classId=' not in file:
            print('"file" path should include a specific class ("classId=XX") !')
            
        else:
            if n == None:
                data = pd.read_parquet(file)

            else:
                all_files = os.listdir(file)
                data = pd.read_parquet(file+all_files[0])

                for i in all_files[1:]:
                    if len(data) < n:
                        data = pd.concat([data, pd.read_parquet(file+i)])

                data = data[:n]
              
            first = file.find('classId=') + len('classId=')
            true_class = file[first:first+3]

            converted = data.apply(ELAsTiCC_lightcurve.convert_ELAsTiCC_format, axis=1)
            lcs = [ELAsTiCC_lightcurve(i) for i in converted]
            for lc in lcs:
                lc.true_class = true_class
                
            return lcs
        
    @staticmethod
    def convert_ELAsTiCC_format(ps):
        prv = ps['prvDiaSources']
        new = ps['diaSource']
        forced = ps['prvDiaForcedSources']
        meta = ps['diaObject']

        time_array = [d['midPointTai'] for d in prv]
        first_detec = min(time_array + [new['midPointTai']])
        detection_duration = new['midPointTai'] - first_detec

        if forced is None:
            non_detec = [], [], [], []
            non_detec_count = None

        # We just want to keep the last forced phot, so that it constrains the rise time but not the color
        else:
            time_array_forced = np.array([d['midPointTai'] for d in forced])
            filter_array_forced = np.array([d['filterName'] for d in forced])
            mask = (time_array_forced < first_detec) & [i in ELAsTiCC_lightcurve.keep_bands for i in filter_array_forced]
            valid = time_array_forced[mask]
            
            if len(valid) != 0:
                last_forced = np.where(time_array_forced==np.max(valid))[0][0]
                non_detec = [forced[last_forced]['midPointTai']], [forced[last_forced]['psFlux']],\
                [forced[last_forced]['psFluxErr']],[forced[last_forced]['filterName']]
                non_detec_count = dict(Counter(non_detec[3]))
                
            else:
                non_detec = [], [], [], []
                non_detec_count = None

        objectId = new['diaObjectId']
        alertId = ps.alertId
        cjd = np.append(new['midPointTai'],  time_array + non_detec[0])
        cflux = np.append(new['psFlux'], [d['psFlux'] for d in prv] + non_detec[1])
        csigflux = np.append(new['psFluxErr'], [d['psFluxErr'] for d in prv] + non_detec[2])
        cfid = np.append(new['filterName'], [d['filterName'] for d in prv] + non_detec[3])

        cjd, cflux, csigflux, cfid = zip(*sorted(zip(cjd, cflux, csigflux, cfid)))
        cjd, cflux, csigflux, cfid = np.array(cjd),np.array(cflux), np.array(csigflux), np.array(cfid)

        additional = {}
        # COMPUTE STATISTICAL FEATURES
        additional['duration'] = detection_duration
        additional['mean_flux'] = np.mean(cflux)
        additional['std_flux'] = np.std(cflux)
        additional['mean_snr'] = np.mean(cflux/csigflux)
        additional['std_snr'] = np.std(cflux/csigflux)

        for band in ELAsTiCC_lightcurve.band_wave_aa:
            additional[f'n_{band}'] = len(cfid[cfid==band])
            additional[f'peak_{band}'] = cflux[cfid==band].max() if len(cflux[cfid==band]!=0) else -999
            
            
        # ELASTICC Meta data
        ra = new['ra']
        dec = new['decl']
        for i in ELAsTiCC_lightcurve.object_metadata:
            additional[i] = meta[i]
            

        band_mask = [i in ELAsTiCC_lightcurve.keep_bands for i in cfid]
        cjd, cflux, csigflux, cfid = cjd[band_mask], cflux[band_mask], csigflux[band_mask], cfid[band_mask]
        return objectId, alertId, cjd, cflux, csigflux, cfid, non_detec_count, additional


class FeatureExtractor():
    def __init__(self, lcs):
        self.feature_names = self.generate_feature_names()
        
        # Use first lc as a standard for metadata requirement
        self.meta_names = list(lcs[0].iloc[0].meta)
        self.features = pd.DataFrame(-999, index=np.arange(len(lcs)),\
                                     columns=['lc', 'type', 'bolometric', 'temperature', 'rising', 'fit_error'] +\
                                     self.feature_names+self.meta_names)
        self.features['lc'] = lcs
        self.features['type'] = self.features['lc'].apply(lambda x: x.true_class)
        
        self.fitting_functions = None
    
    def find_fitfunc(self):
        self.check_points()
        self.features = self.features.apply(self.apply_fitfunc, axis=1,)

    @staticmethod
    def generate_feature_names():
        
        # Loop over the function parameters.
        functions = [blm.ExpBolometricTerm, blm.SigmoidBolometricTerm,
             blm.LinexpBolometricTerm, blm.BazinBolometricTerm,
            blm.DoublexpBolometricTerm, tp.ConstantTemperatureTerm, tp.SigmoidTemperatureTerm]
        
        function_names = ['exp', 'sigmoid', 'linexp', 'bazin', 'doublexp', 'constant', 'Tsigmoid']
        feature_names = []
        for i in range(len(functions)):
            feature_names += [param + '_' + function_names[i] for param in functions[i].parameter_names()]
        return feature_names

    @staticmethod
    def apply_fitfunc(pds):
        
        if len(pds['lc'].flux)>0:
            bband, bcol = pds['lc'].biggest_band,  pds['lc'].biggest_color

            if bband >= 6:
                pds['bolometric'] = 'doublexp'
            elif bband == 5:
                pds['bolometric'] = 'bazin'
            elif bband == 4:
                pds['bolometric'] = 'linexp'
            elif bband == 3:
                pds['bolometric'] = 'exp'
                pds['lc'].mjd -= np.mean(pds['lc'].mjd)
            else:
                pds['bolometric'] = None

            # If lightcurve is purely rising or falling limit ourselves to sigmoid fit
            if (bband >= 4) & (pds['lc'].rising != 0):
                pds['bolometric'] = 'sigmoid'

            if bcol == None:
                pds['temperature'] = 'colorless'
            # Because of some degeneracies between pseudo amplitude and t0 we must be limited to constant temp
            elif (bcol >= 3) and (pds['bolometric'] != 'exp'):
                pds['temperature'] = 'Tsigmoid'
            elif bcol >= 1:
                pds['temperature'] = 'constant'           
                
        else:
            pds['bolometric'] = None
            pds['temperature'] = None
        
        return pds
    
    def fit_rainbow(self):

        # Filter out ExperimentalWarnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self.features = self.features.apply(self.apply_rainbow, axis=1)
        
    @staticmethod
    def apply_rainbow(pds):

        if pds['bolometric'] != None:
            fitter = RainbowFit.from_angstrom(pds['lc'].band_wave_aa, with_baseline=False,\
                                       temperature=pds['temperature'], bolometric=pds['bolometric'])#, 

            try:
                result = fitter(pds['lc'].mjd, pds['lc'].flux, sigma=pds['lc'].fluxerr, band=pds['lc'].band)
                for idx, name in enumerate(fitter.names):
                    function_name = pds['bolometric'] if idx < len(fitter.bolometric.parameter_names()) else pds['temperature']
                    pds[name + '_' + function_name] = result[idx]

                pds['fit_error'] = result[-1]
                return pds

            except RuntimeError:
                return pds

        else:
            return pds

    def get_metadata(self):
        for i in self.meta_names:
            self.features[i] = self.features['lc'].apply(lambda x: x.meta[i])
    
    def full_feature_extraction(self):
        self.get_metadata()
        self.find_fitfunc()
        self.fit_rainbow()

    @staticmethod  
    def plot_fit(ps, xlim=None, ylim=None):
        lc = ps['lc']        
        model = RainbowFit.from_angstrom(lc.band_wave_aa, with_baseline=False,\
                                         temperature=ps['temperature'], bolometric=ps['bolometric'])

            
        function_name = [ps['bolometric'] if idx < len(model.bolometric.parameter_names()) else ps['temperature'] for            idx, name in enumerate(model.names)]
        new_names = [name + '_' + function_name[idx] for idx, name in enumerate(model.names)]
        params = [ps[name] for name in new_names] + [0.]
        lc.plot(xlim=xlim, ylim=ylim, fit=[model.model, params])
        return dict(zip(new_names, params[:-1]))
        
    def check_points(self):
        self.features['lc'].apply(lambda x: x.count_points())
        
    def format_all(self):
        self.features['lc'].apply(lambda x: x.full_format())
        self.features['rising'] = self.features['lc'].apply(lambda x: x.rising)
        

        

