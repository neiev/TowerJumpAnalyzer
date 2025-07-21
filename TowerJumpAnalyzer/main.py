# 
from typing import Optional
import pandas as pd
import numpy as np
import csv
import json
import math
import os
import re
from datetime import datetime
import traceback
from shapely.geometry import Point, shape

class TowerJumpAnalyzer:
    def __init__(self):
        # Configurações principais
        self.min_time_diff_threshold = 5 * 60  # 5 minutos em segundos
        self.max_time_diff_threshold = 60 * 60  # 1 hora em segundos
        self.max_speed_threshold = 900  # max km/h para considerar um tower jump
        self.max_jump_distance = 300  # Distância máxima para considerar um tower jump (em km)
        self.min_jump_distance = 10  # Distância mínima para considerar um tower jump (em km)
        self.date_format = '%m/%d/%y %H:%M'
        self.min_confidence = 0

        # Limiares para resolução de conflitos
        self.min_duration_to_override = 3  # minutos
        self.min_confidence_diff = 20  # porcentagem
        self.min_confidence_absolute = 70  # porcentagem mínima

    def safe_float(self, value):
        """Converte string para float de forma segura"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    def haversine_distance_vectorized(self, lon1, lat1, lon2, lat2):
        """
        Calcula a distância entre dois pares de coordenadas (em vetores) usando a fórmula de Haversine.
        Entradas e saídas em quilômetros.
        """

        # Converter graus para radianos
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        R = 6371  # Raio da Terra em quilômetros
        return R * c

        
    def is_valid_record(self, record):
        """Verifica se o registro é válido para análise"""
    
        try:          
            if record is None:
                return False
        
            if 'start_time' not in record or 'end_time' not in record:
                return False

            if not isinstance(record['start_time'], datetime) or not isinstance(record['end_time'], datetime):
                return False

            if record['state'] == 'UNKNOWN':
                return False

            if record['latitude'] is None or record['longitude'] is None:
                return False

            if record['latitude'] == 0 and record['longitude'] == 0:
                return False

            if record['latitude'] <= -90 and record['longitude'] <= -180:
                return False
            
            return True
            
        except Exception as e:
            print(record)
            print(f"Erro ao verificar validade do registro: {e}")
            traceback.print_exc()
            return False

    def detect_vehicle_type(self, speed, distance, time_diff):
        # if distance == 0:
        # #     return 'UNKNOWN'
        
        vehicle_types = {
            'Andando': (0, 7),
            'Bicicleta': (7, 30),
            'Carro': (20, 200),
            'Ônibus': (10, 120),
            'Barco': (5, 100),
            'Trem': (30, 300),
            'Avião': (200, 900),
        }
        
        for vehicle, (min_speed, max_speed) in vehicle_types.items():
            if min_speed <= speed <= max_speed:
                if time_diff > 60 and (speed * (time_diff / 3600)) >= distance:
                    return vehicle
                    
        return 'UNKNOWN'

    def detect_jumps(self, df):
        if df.empty:
            return df

        MAX_TIME_DIFF = 3600  # 1h em segundos

        # ---------------------------------------------
        # 1. Estado válido
        # ---------------------------------------------
        df['is_state_valid'] = (
            (df['state'] != 'UNKNOWN') &
            df['latitude'].notna() & (df['latitude'] != 0) &
            df['longitude'].notna() & (df['longitude'] != 0)
        )

        # ---------------------------------------------
        # 2. Colunas de tempo e comparação
        # ---------------------------------------------
        df['prev_state'] = df['state'].shift(1).fillna('UNKNOWN')
        df['prev_end_time'] = df['end_time'].shift(1)
        df['prev_lat'] = df['latitude'].shift(1)
        df['prev_lon'] = df['longitude'].shift(1)
        df['prev_is_valid'] = df['is_state_valid'].shift(1).fillna(False)

        df['time_diff'] = (df['start_time'] - df['prev_end_time']).dt.total_seconds()
        df['state_changed'] = df['state'] != df['prev_state']
        df['valid_state_change'] = df['state_changed'] & df['is_state_valid'] & df['prev_is_valid']
        df['valid_state_change_1h'] = df['valid_state_change'] & (df['time_diff'] <= MAX_TIME_DIFF)

        # ---------------------------------------------
        # 3. Cálculo vetorizado de distância e velocidade
        # ---------------------------------------------
        valid_movement_mask = df['is_state_valid'] & df['prev_is_valid'] & (df['time_diff'] > 0)

        df['distance'] = 0.0
        df['speed'] = 0.0
        df.loc[valid_movement_mask, 'distance'] = self.haversine_distance_vectorized(
            df.loc[valid_movement_mask, 'prev_lon'],
            df.loc[valid_movement_mask, 'prev_lat'],
            df.loc[valid_movement_mask, 'longitude'],
            df.loc[valid_movement_mask, 'latitude']
        )
        df.loc[valid_movement_mask, 'speed'] = (
            df.loc[valid_movement_mask, 'distance'] * 3600 / df.loc[valid_movement_mask, 'time_diff']
        )

        df['is_movement_possible'] = (
            (df['distance'] > 0) &
            ((df['speed'] * (df['time_diff'] / 3600)) >= df['distance'])
        )

        # ---------------------------------------------
        # 4. Inicializar colunas auxiliares
        # ---------------------------------------------
        df['tower_jump'] = False
        df['conflict_resolution'] = 'NO_CONFLICT'
        df['discarded_state'] = None
        df['resolved_by'] = None
        df['older_state'] = None
        df['older_state_index'] = None

        # ---------------------------------------------
        # 5. Loop apenas onde há mudança válida de estado
        # ---------------------------------------------
        candidates = df[df['valid_state_change_1h'] & (~df['is_movement_possible'])]

        for i in candidates.index:
            row = df.loc[i]
            base_state = row['prev_state']
            current_time = row['start_time']

            # Busca de estado histórico diferente (até 1h antes)
            older_idx = None
            for j in range(i - 1, max(i - 1000, -1), -1):
                candidate = df.loc[j]
                if not candidate['is_state_valid']:
                    continue
                if abs((candidate['start_time'] - current_time).total_seconds()) > MAX_TIME_DIFF:
                    break
                if candidate['state'] != base_state:
                    older_idx = j
                    break

            historical_divergence = (
                older_idx is not None and
                row['state'] != df.at[older_idx, 'state']
            )

            if older_idx is None or historical_divergence:
                df.at[i, 'tower_jump'] = True
                df.at[i, 'conflict_resolution'] = 'STATE_DIVERGENCE'
                df.at[i, 'discarded_state'] = base_state
                df.at[i, 'resolved_by'] = 'DISTANCE_TIME_ANALYSIS'
                df.at[i, 'older_state_index'] = older_idx
                df.at[i, 'older_state'] = df.at[older_idx, 'state'] if older_idx is not None else None

                # Verificação do estado posterior
                if i + 1 < len(df) and df.at[i + 1, 'state'] == base_state:
                    df.at[i, 'conflict_resolution'] = 'STATE_RECOVERY'

        return df

    def load_data(self, file_path):
        """Carrega e processa os dados do arquivo CSV usando pandas"""
        try:
            # Carrega o CSV com pandas
            df = pd.read_csv(file_path, parse_dates=['UTCDateTime'], date_format='%m/%d/%y %H:%M')
            
            required_columns = {'UTCDateTime', 'LocalDateTime', 'State', 'Latitude', 'Longitude'}
            if not all(col in df.columns for col in required_columns):
                missing = required_columns - set(df.columns)
                print(f"Erro: Colunas obrigatórias faltando: {missing}")
                return pd.DataFrame()

            df[['State']] = df[['State']].fillna('UNKNOWN')
            df[['Latitude','Longitude']] =  df[['Latitude','Longitude']].fillna('0.0')

            # Processa os dados
            processed_data = []
           
            for index, row in df.iterrows():
                try:
                    start_time = row.get('UTCDateTime')

                    # Processa latitude e longitude
                    unformatted_lat = row.get('Latitude')
                    if unformatted_lat is not None and isinstance(unformatted_lat, str):
                        unformatted_lat = unformatted_lat[:8]
                    
                    unformatted_lon = row.get('Longitude')
                    if isinstance(unformatted_lon, str):
                        unformatted_lon = unformatted_lon[:9]
                    
                    lat = self.safe_float(unformatted_lat)
                    lon = self.safe_float(unformatted_lon)
                    
                    # Determina o estado
                    state = row.get('State', 'UNKNOWN')

                    # Cria a entrada de dados
                    entry = {
                        'start_time': start_time,
                        'end_time': start_time,
                        'latitude': lat,
                        'longitude': lon,
                        'state': state,
                        'discarded_state': None,
                        'confidence': 100,
                        'same_time_diff_state': False,
                        'duration': 0,
                        'speed': 0,
                        'distance': 0,
                        'vehicle_type': 'UNKNOWN',
                        'movement_possible': False,
                        'tower_jump': False,
                        'conflict_resolution': 'NO_CONFLICT',
                        'resolved_by': None,
                        'location_score': 0                        
                    }
                    
                    processed_data.append(entry)
                
                except Exception as e:
                    print(f"Erro ao processar linha {index+1}: {str(e)}")
                    continue
            
            # Cria DataFrame com os dados processados
            result_df = pd.DataFrame(processed_data)
            
            if result_df.empty:
                print("Aviso: O arquivo foi lido, mas nenhum dado válido foi encontrado")
                return pd.DataFrame()
            
            # Ordena por tempo
            result_df = result_df.sort_values('start_time')
            
            print(f"Carregados {len(result_df)} registros válidos")
            
            unknown_count = sum(1 for state in result_df['state'] if state == 'UNKNOWN')
            if unknown_count > 0:
                print(f"AVISO: {unknown_count} registros ({unknown_count/len(result_df):.2%}) com estado UNKNOWN")
                if unknown_count/len(result_df) > 0.05:
                    print("  Ação recomendada: Verificar qualidade das coordenadas e mapeamento de estados")
            
            return result_df
        
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Erro inesperado ao ler arquivo: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def format_distance(self, distance):
        """Formata a distância em km com 2 casas decimais"""
        try:
            if distance is None or np.isnan(distance):
                return ''
            return f"{distance:.2f} km" if distance >= 0 else ''
        except (ValueError, TypeError) as e:
            print(f"Erro ao formatar distância: {str(e)}")
            traceback.print_exc()
            return ''

    def format_speed(self, speed):
        """Formata a velocidade em km/h com 2 casas decimais"""
        if speed is None or np.isnan(speed):
            return ''
        return f"{speed:.2f} km/h" if speed >= 0 else ''

    def generate_report(self, df, output_file):
        """Gera um relatório CSV com os dados processados"""
        if df.empty:
            print("Nenhum dado para gerar relatório")
            return False

        try:
            # Formata as colunas de data/hora
            df['start_time_str'] = df['start_time'].dt.strftime(self.date_format)
            df['end_time_str'] = df['end_time'].dt.strftime(self.date_format)
            
            # Formata as colunas numéricas
            df['speed_str'] = df['speed'].apply(self.format_speed)
            df['distance_str'] = df['distance'].apply(self.format_distance)

            # Prepara o DataFrame para exportação
            export_df = df[[
                'start_time_str', 'end_time_str', 'latitude', 'longitude',
                'state', 'discarded_state',
                'duration', 'speed_str', 'distance_str', 'vehicle_type', 
                'movement_possible', 'tower_jump', 'conflict_resolution', 
                'resolved_by', 'location_score', 'confidence'
            ]]
            
            # Renomeia as colunas
            export_df = export_df.rename(columns={
                'start_time_str': 'start_time',
                'end_time_str': 'end_time',
                'speed_str': 'speed',
                'distance_str': 'distance'
            })
            
            # Exporta para CSV
            export_df.to_csv(output_file, index=False)
            return True
            
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            traceback.print_exc()
            return False

    def print_summary(self, df):
        """Imprime um resumo dos dados processados"""
        if df.empty:
            print("Nenhum dado para resumo")
            return

        total_records = len(df)
        tower_jump_count = df['tower_jump'].sum()
        unknown_state_count = df['state'].eq('UNKNOWN').sum()

        print("\n=== RESUMO DOS DADOS ===")
        print(f"Total de registros: {total_records}")
        print(f"Registros com Tower Jump: {tower_jump_count} ({tower_jump_count/total_records:.2%})")
        print(f"Registros com estado UNKNOWN: {unknown_state_count} ({unknown_state_count/total_records:.2%})")

def main():
    print("=== Tower Jump Analyzer ===")
    print("Sistema avançado de análise de localização e detecção de tower jumps usando pandas\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "localization")
    input_file = os.path.join(data_dir, "CarrierData.csv")
    output_file_name = 'TowerJumpReport.csv'
    # output_file_name = 'main_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    output_file = os.path.join(base_dir, output_file_name)

    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada não encontrado em {input_file}")
        print("Certifique-se que o arquivo CarrierData.csv está na pasta 'localization'")
        return

    analyzer = TowerJumpAnalyzer()
    print("Carregando e processando dados...")
    df = analyzer.load_data(input_file)

    if df.empty:
        print("Nenhum dado válido encontrado no arquivo de entrada.")
        return

    print("Detectando tower jumps...")
    df = analyzer.detect_jumps(df)

    # print("Resolvendo conflitos de estado...")
    # df = analyzer.resolve_state_conflicts(df)

    # print("Obtendo localização atual...")
    # current_location = analyzer.get_current_location(df)

    print("Gerando relatório...")
    if analyzer.generate_report(df, output_file):
        print(f"\nRelatório gerado com sucesso em: {output_file}")
    else:
        print("Falha ao gerar relatório.")
        return

    analyzer.print_summary(df)

if __name__ == "__main__":
    main()