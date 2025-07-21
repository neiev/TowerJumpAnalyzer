# 
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

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcula a distância em km entre dois pontos geográficos usando a fórmula de Haversine"""
        if None in (lat1, lon1, lat2, lon2):
            return 0
            
        # Converte coordenadas para radianos
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Fórmula de Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Raio da Terra em quilômetros
        return c * r

    def check_consistency(self, record, data, index):
        """Verifica se há registros consistentes subsequentes"""
        # Verificar próximos 3 registros
        consistent_count = 0
        for j in range(index + 1, min(index + 4, len(data))):
            next_rec = data[j]
            if not self.is_valid_record(next_rec):
                continue

            # Verificar mesma localização e estado
            same_location = self.haversine_distance(
                record['lat'], record['lon'],
                next_rec['lat'], next_rec['lon']
            ) < 1

            same_state = record['state'] == next_rec['state']

            if same_location and same_state:
                consistent_count += 1

        # Verificar 3 registros anteriores
        for j in range(index - 1, max(index - 4, -1), -1):
            prev_rec = data[j]
            if not self.is_valid_record(prev_rec):
                continue

            # Verificar mesma localização e estado
            same_location = self.haversine_distance(
                record['lat'], record['lon'],
                prev_rec['lat'], prev_rec['lon']
            ) < 1

            same_state = record['state'] == prev_rec['state']

            if same_location:
                consistent_count -= 2
            if same_state:
                consistent_count -= 2

        return consistent_count >= 2  # Pelo menos 2 registros consistentes

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calcula a distância entre dois pontos na Terra usando a fórmula de Haversine.
        
        Args:
            lat1, lon1: Latitude e longitude do primeiro ponto (em graus)
            lat2, lon2: Latitude e longitude do segundo ponto (em graus)
            
        Returns:
            float: Distância em quilômetros entre os dois pontos
        """
        # Converte graus para radianos
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Fórmula de Haversine
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Raio da Terra em quilômetros
        
        # Retorna a distância em quilômetros
        return c * r

    # def detect_vehicle_type(self, speed, distance, time_diff):
    #     """
    #     Detecta o tipo provável de veículo com base na velocidade de deslocamento.
        
    #     Args:
    #         speed: Velocidade média em km/h
    #         distance: Distância percorrida em km
    #         time_diff: Diferença de tempo em segundos
            
    #     Returns:
    #         str: Tipo de veículo provável (AIRPLANE, CAR, TRAIN, WALKING, STATIONARY)
    #     """
    #     # Se a distância ou tempo for zero, não há movimento
    #     if distance < 0.1 or time_diff < 60:
    #         return "STATIONARY"
        
    #     # Classifica o tipo de veículo com base na velocidade
    #     if speed > 400:
    #         return "AIRPLANE"
    #     elif speed > 180:
    #         return "HIGH_SPEED_TRAIN"
    #     elif speed > 80:
    #         return "TRAIN"
    #     elif speed > 5:
    #         return "CAR"
    #     elif speed > 0.5:
    #         return "WALKING"
    #     else:
    #         return "UNKNOWN"
        
    def is_valid_record(self, record):
        """Verifica se o registro é válido para análise"""
    
        try:          
            if record is None:
                return False
        
            if 'start_time' not in record or 'end_time' not in record:
                return False

            if not isinstance(record['start_time'], datetime) or not isinstance(record['end_time'], datetime):
                return False

            # if 'latitude' not in record or 'longitude' not in record:
            #     return False

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

    def calcular_distancia_e_velocidade(self, df):
        # Preparar dados deslocados
        df_shift = df.shift(1)
        valid_mask = (~df_shift['latitude'].isna()) & (~df_shift['longitude'].isna())
        
        # Calcular diferença de tempo
        time_diff = (df['start_time'] - df_shift['end_time']).dt.total_seconds()
        
        # Calcular distância
        distance = np.zeros(len(df))
        distance[1:] = self.haversine_distance(
            df_shift['longitude'].values[1:],
            df_shift['latitude'].values[1:],
            df['longitude'].values[1:],
            df['latitude'].values[1:]
        )
        
        # Calcular velocidade
        speed = np.zeros(len(df))
        valid_time = time_diff > 0
        speed[1:][valid_time[1:]] = (distance[1:][valid_time[1:]] * 3600) / time_diff[1:][valid_time[1:]]
        
        return distance, speed

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

        # Pré-processamento: criar colunas de estado válido
        valid_mask = (df['state'] != 'UNKNOWN') & \
                    (~df['latitude'].isna()) & \
                    (~df['longitude'].isna()) & \
                    (df['latitude'] != 0) & \
                    (df['longitude'] != 0)
        
        # Identificar mudanças de estado
        prev_state = df['state'].shift(1).fillna('UNKNOWN')
        same_time = df['start_time'] == df['start_time'].shift(1)
        same_state = df['state'] == prev_state
        state_unknown = df['state'] == 'UNKNOWN'
        prev_unknown = df['state'].shift(1) == 'UNKNOWN'
        
        conditions = [~same_state & same_time & ~state_unknown & ~prev_unknown]
        choices = [
            True
        ]
        df['same_time_diff_state'] = np.select(conditions, choices, default='')
                
        # Inicializar colunas de resultados
        df['tower_jump'] = False
        df['conflict_resolution'] = 'NO_CONFLICT'
        df['discarded_state'] = None
        df['resolved_by'] = None
        df['prev_valid_index'] = None
        df['next_valid_index'] = None
        df['prev_valid_state'] = None
        df['next_valid_state'] = None
        df['is_movement_possible'] = False

        # Para cada registro, encontrar estados válidos anterior e posterior
        for i in range(len(df)):
            current = df.iloc[i]
            
            # Ignorar registros inválidos
            if not self.is_valid_record(current):
                continue
                
            # Buscar estado válido anterior diferente deste
            prev_index = i - 1
            while prev_index >= 0:
                prev_rec = df.iloc[prev_index]
                if self.is_valid_record(prev_rec) and \
                   prev_rec['state'] != 'UNKNOWN' and \
                   prev_rec['state'] != current['state']:
                    break
                prev_index -= 1
                
            # Buscar estado válido posterior diferente do atual
            next_index = i + 1
            while next_index < len(df):
                next_rec = df.iloc[next_index]
                if self.is_valid_record(next_rec) and \
                   next_rec['state'] != 'UNKNOWN' and \
                   next_rec['state'] != current['state']:
                    break
                next_index += 1
                
            # Atualizar DataFrame com índices encontrados
            df.at[i, 'prev_valid_index'] = prev_index if prev_index >= 0 else None
            df.at[i, 'next_valid_index'] = next_index if next_index < len(df) else None
            
            # Processar apenas se encontrou estado válido anterior
            if prev_index >= 0:
                prev_rec = df.iloc[prev_index]
                df.at[i, 'prev_valid_state'] = prev_rec['state']
                df.at[i, 'prev_valid_state_start_time'] = prev_rec['start_time']
                
                # Calcular distância e tempo
                distance = self.haversine_distance(
                    prev_rec['longitude'], prev_rec['latitude'],
                    current['longitude'], current['latitude']
                )
                time_diff = (current['start_time'] - prev_rec['end_time']).total_seconds()
                
                # Calcular velocidade
                speed = (distance * 3600) / time_diff if time_diff > 0 else 0
                df.at[i, 'distance'] = distance
                df.at[i, 'speed'] = speed
                df.at[i, 'duration'] = time_diff
                
                # Verificar se o movimento é possível
                is_movement_possible = time_diff > 60 and distance > 0 and (speed * (time_diff / 3600)) >= distance
                df.at[i, 'is_movement_possible'] = is_movement_possible
                # Detectar tipo de veículo
                df.at[i, 'vehicle_type'] = self.detect_vehicle_type(speed, distance, time_diff)

                # Verificar divergência de estado
                state_divergence = current['state'] != prev_rec['state']
                
                # Se estados divergem e movimento não é possível, marcar como tower jump
                if state_divergence and not is_movement_possible:
                    df.at[i, 'tower_jump'] = True
                    df.at[i, 'conflict_resolution'] = 'STATE_DIVERGENCE'
                    df.at[i, 'discarded_state'] = prev_rec['state']
                    df.at[i, 'resolved_by'] = 'DISTANCE_TIME_ANALYSIS'
                    
                    # Verificar consistência com estado posterior
                    if next_index < len(df):
                        next_rec = df.iloc[next_index]
                        df.at[i, 'next_valid_state'] = next_rec['state']
                        
                        # Se próximo estado volta ao estado anterior, confirma tower jump
                        if next_rec['state'] == prev_rec['state']:
                            df.at[i, 'conflict_resolution'] = 'STATE_RECOVERY'
            
            # Processar se encontrou estado válido posterior mas não anterior
            elif next_index < len(df):
                next_rec = df.iloc[next_index]
                df.at[i, 'next_valid_state'] = next_rec['state']
                
                # Calcular distância e tempo para o próximo válido
                distance = self.haversine_distance(
                    current['longitude'], current['latitude'],
                    next_rec['longitude'], next_rec['latitude']
                )
                time_diff = (next_rec['start_time'] - current['end_time']).total_seconds()
                
                # Calcular velocidade
                speed = (distance * 3600) / time_diff if time_diff > 0 else 0
                
                # Verificar se o movimento é possível
                is_movement_possible = (speed * (time_diff / 3600)) >= distance

                # Se movimento não é possível, marcar como tower jump
                if not is_movement_possible:
                    df.at[i, 'tower_jump'] = True
                    df.at[i, 'conflict_resolution'] = 'STATE_DIVERGENCE_FORWARD'
                    df.at[i, 'discarded_state'] = next_rec['state']
                    df.at[i, 'resolved_by'] = 'DISTANCE_TIME_ANALYSIS'

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
                'state', 'discarded_state', 'prev_valid_state', 'next_valid_state',
                'same_time_diff_state', 
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