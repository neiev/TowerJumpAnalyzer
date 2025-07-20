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
            return None

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

    def marcar_tower_jump(self, current, previous):
        """Marca um registro como tower jump"""
        current['tower_jump'] = True
        current['conflict_resolution'] = 'TOWER_JUMP'
        return current

    def check_consistency(self, current, data, index):
        """Verifica a consistência do estado atual com registros próximos"""
        # Implementação simplificada
        return True

    # def detect_jumps(self, df):
    #     """Detecta tower jumps usando pandas DataFrame"""
    #     if df.empty:
    #         return df

    #     # Ordena por tempo
    #     df = df.sort_values('start_time')
        
    #     # Calcula diferenças entre registros consecutivos
    #     df['prev_latitude'] = df['latitude'].shift(1)
    #     df['prev_longitude'] = df['longitude'].shift(1)
    #     df['prev_state'] = df['state'].shift(1)
    #     df['prev_time'] = df['start_time'].shift(1)
        
    #     # Calcula distância e tempo
    #     df['time_diff'] = (df['start_time'] - df['prev_time']).dt.total_seconds()
        
    #     # Função para calcular distância
    #     def calc_distance(row):
    #         return self.calculate_distance(
    #             row['prev_latitude'], row['prev_longitude'],
    #             row['latitude'], row['longitude']
    #         )
        
    #     df['distance'] = df.apply(calc_distance, axis=1)
        
    #     # Calcula velocidade em km/h
    #     df['speed'] = df.apply(
    #         lambda row: (row['distance'] / (row['time_diff']/3600)) if row['time_diff'] > 0 else 0, 
    #         axis=1
    #     )
        
    #     # Verifica possibilidade de mudança de local
    #     df['is_location_change_possible'] = df.apply(
    #         lambda row: (row['speed'] * (row['time_diff'] / 3600)) >= row['distance'] 
    #         if row['distance'] > 0 and row['time_diff'] > 0 else True,
    #         axis=1
    #     )
        
    #     # Verifica saltos entre estados
    #     df['is_same_state'] = df['state'] == df['prev_state']
        
    #     # Verifica estados simultâneos
    #     def determine_time_diff_state(row):
    #         if row['time_diff'] is None or np.isnan(row['time_diff']):
    #             return 'UNKNOWN'
    #         elif row['time_diff'] == 0:
    #             return 'SAME_TIME_DIFF_STATE'
    #         else:
    #             return 'DIFF_TIME_DIFF_STATE'
        
    #     df['same_time_diff_state'] = df.apply(determine_time_diff_state, axis=1)
        
    #     # Inicializa colunas de resolução de conflitos
    #     df['conflict_resolution'] = 'NO_CONFLICT'
    #     df['tower_jump'] = False
    #     df['discarded_state'] = None
    #     df['resolved_by'] = None
        
    #     # Detecta tower jumps
    #     for i, row in df.iterrows():
    #         if i == 0:  # Ignora o primeiro registro
    #             continue
                
    #         if not row['is_same_state'] and not row['is_location_change_possible']:
    #             df.at[i, 'conflict_resolution'] = 'LOCATION_CHANGE_IMPOSSIBLE'
    #             df.at[i, 'tower_jump'] = True
    #             continue
                
    #         # 1. Registros simultâneos em estados diferentes
    #         if (not row['is_same_state'] and 
    #             not row['same_time_diff_state'] == 'SAME_TIME_DIFF_STATE'):
    #             df.at[i, 'conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
    #             is_consistent = self.check_consistency(row, df, i)
    #             if not is_consistent:
    #                 df.at[i, 'conflict_resolution'] = 'DIFF_STATE_JUMP'
    #                 df.at[i, 'tower_jump'] = True
    #             continue
                
    #         # 2. Registros com estado diferente e tempo de diferença curto
    #         elif (not row['is_same_state'] and
    #               row['same_time_diff_state'] == 'DIFF_TIME_DIFF_STATE' and
    #               row['time_diff'] < self.min_time_diff_threshold):
    #             df.at[i, 'conflict_resolution'] = 'DIFF_STATE_SHORT_TIME'
    #             is_consistent = self.check_consistency(row, df, i)
    #             if not is_consistent and not row['is_location_change_possible']:
    #                 df.at[i, 'tower_jump'] = True
    #             continue
        
    #     # Remove colunas temporárias
    #     df = df.drop(['prev_latitude', 'prev_longitude', 'prev_state', 'prev_time', 
    #                   'is_same_state'], axis=1)
        
    #     return df

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

    def detect_vehicle_type(self, speed, distance, time_diff):
        """
        Detecta o tipo provável de veículo com base na velocidade de deslocamento.
        
        Args:
            speed: Velocidade média em km/h
            distance: Distância percorrida em km
            time_diff: Diferença de tempo em segundos
            
        Returns:
            str: Tipo de veículo provável (AIRPLANE, CAR, TRAIN, WALKING, STATIONARY)
        """
        # Se a distância ou tempo for zero, não há movimento
        if distance < 0.1 or time_diff < 60:
            return "STATIONARY"
        
        # Classifica o tipo de veículo com base na velocidade
        if speed > 400:
            return "AIRPLANE"
        elif speed > 180:
            return "HIGH_SPEED_TRAIN"
        elif speed > 80:
            return "TRAIN"
        elif speed > 5:
            return "CAR"
        elif speed > 0.5:
            return "WALKING"
        else:
            return "UNKNOWN"
    
    def detect_jumps(self, df):
        """
        Detecta tower jumps nos dados, comparando estados consecutivos e calculando
        distâncias e velocidades entre pontos.
        
        Args:
            df: DataFrame com os dados a serem analisados
            
        Returns:
            DataFrame: DataFrame original com colunas adicionais de análise
        """
        if df.empty:
            return df
        
        # Ordena por timestamp
        df = df.sort_values(by=['start_time'])
        
        # Inicializa colunas de análise
        df['prev_state'] = df['state'].shift(1)
        df['prev_latitude'] = df['latitude'].shift(1)
        df['prev_longitude'] = df['longitude'].shift(1)
        df['prev_start_time'] = df['start_time'].shift(1)
        df['time_diff'] = (df['start_time'] - df['prev_start_time']).dt.total_seconds()
        df['tower_jump'] = False
        df['distance'] = None
        df['speed'] = None
        df['vehicle_type'] = None
        df['is_location_change_possible'] = None
        
        # Processa cada linha
        for index, row in df.iterrows():
            # Pula a primeira linha que não tem 'anterior'
            if pd.isna(row['prev_start_time']):
                continue
            
            # Verifica se há mudança de estado ou se estado anterior é inválido
            if row['state'] != row['prev_state'] or row['prev_state'] == 'UNKNOWN' or \
            pd.isna(row['prev_latitude']) or pd.isna(row['prev_longitude']) or \
            row['prev_latitude'] == 0 or row['prev_longitude'] == 0:
                
                # Se o estado atual também for inválido, continue
                if row['state'] == 'UNKNOWN' or pd.isna(row['latitude']) or pd.isna(row['longitude']) or \
                row['latitude'] == 0 or row['longitude'] == 0:
                    continue
                
                # Encontra o último estado válido anterior
                valid_prev = None
                for prev_idx in range(index-1, -1, -1):
                    prev_row = df.iloc[prev_idx]
                    if prev_row['state'] != 'UNKNOWN' and not pd.isna(prev_row['latitude']) and \
                    not pd.isna(prev_row['longitude']) and prev_row['latitude'] != 0 and \
                    prev_row['longitude'] != 0:
                        valid_prev = prev_row
                        break
                
                if valid_prev is not None:
                    # Calcula distância entre pontos válidos
                    try:
                        distance = self.haversine_distance(
                            valid_prev['latitude'], valid_prev['longitude'],
                            row['latitude'], row['longitude']
                        )
                        
                        # Calcula o tempo entre registros válidos
                        time_diff = (row['start_time'] - valid_prev['start_time']).total_seconds()
                        
                        # Calcula velocidade
                        if time_diff > 0:
                            speed = distance / (time_diff / 3600)  # km/h
                        else:
                            speed = 0
                        
                        # Determina o tipo de veículo
                        vehicle_type = self.detect_vehicle_type(speed, distance, time_diff)
                        
                        # Verifica se o deslocamento é possível no tempo dado
                        is_possible = self.is_location_change_possible(distance, time_diff, speed)
                        
                        # Atualiza o DataFrame
                        df.at[index, 'distance'] = distance
                        df.at[index, 'speed'] = speed
                        df.at[index, 'vehicle_type'] = vehicle_type
                        df.at[index, 'is_location_change_possible'] = is_possible
                        
                        # Define tower_jump se não for possível o deslocamento
                        if not is_possible:
                            df.at[index, 'tower_jump'] = True
                            
                    except Exception as e:
                        print(f"Erro ao calcular distância: {e}")
                        traceback.print_exc()
            
            # Se não houve mudança de estado, verifica outros critérios para tower jump
            elif not pd.isna(row['prev_latitude']) and not pd.isna(row['prev_longitude']) and \
                row['prev_latitude'] != 0 and row['prev_longitude'] != 0:
                try:
                    # Calcula distância entre pontos consecutivos do mesmo estado
                    distance = self.haversine_distance(
                        row['prev_latitude'], row['prev_longitude'],
                        row['latitude'], row['longitude']
                    )
                    
                    # Se a distância for grande mesmo no mesmo estado, pode ser tower jump
                    if distance > self.min_jump_distance and row['time_diff'] > 0:
                        speed = distance / (row['time_diff'] / 3600)  # km/h
                        vehicle_type = self.detect_vehicle_type(speed, distance, row['time_diff'])
                        is_possible = self.is_location_change_possible(distance, row['time_diff'], speed)
                        
                        df.at[index, 'distance'] = distance
                        df.at[index, 'speed'] = speed
                        df.at[index, 'vehicle_type'] = vehicle_type
                        df.at[index, 'is_location_change_possible'] = is_possible
                        
                        if not is_possible:
                            df.at[index, 'tower_jump'] = True
                except Exception as e:
                    print(f"Erro ao verificar tower jump no mesmo estado: {e}")
                    traceback.print_exc()
        
        return df  
    
    def is_location_change_possible(self, distance, time_diff, speed):
        """
        Verifica se é fisicamente possível o deslocamento dado a distância e o tempo.
        
        Args:
            distance: Distância em km
            time_diff: Diferença de tempo em segundos
            speed: Velocidade calculada em km/h
            
        Returns:
            bool: True se o deslocamento é possível, False caso contrário
        """
        # Calcula a distância máxima que poderia ser percorrida com a velocidade dada
        # no intervalo de tempo disponível
        max_possible_distance = speed * (time_diff / 3600)
        
        # Se a distância máxima possível é maior ou igual à distância real,
        # então o deslocamento é possível
        return max_possible_distance >= distance
   

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
            
            # Processa os dados
            processed_data = []
            for index, row in df.iterrows():
                try:
                    start_time = row.get('UTCDateTime')

                    # Processa latitude e longitude
                    unformatted_lat = row.get('Latitude')
                    if isinstance(unformatted_lat, str):
                        unformatted_lat = unformatted_lat[:8]
                    
                    unformatted_lon = row.get('Longitude')
                    if isinstance(unformatted_lon, str):
                        unformatted_lon = unformatted_lon[:9]
                    
                    lat = self.safe_float(unformatted_lat)
                    lon = self.safe_float(unformatted_lon)
                    
                    # Determina o estado
                    state = row.get('State', 'UNKNOWN')
                    if state == '':
                        state = 'UNKNOWN'
                    
                    # Cria a entrada de dados
                    entry = {
                        'start_time': start_time,
                        'end_time': start_time,
                        'state': state,
                        'confidence': 100,
                        'latitude': lat,
                        'longitude': lon,
                        'duration': 0,
                        'tower_jump': False,
                        'same_time_diff_state': False,
                        'cell_types': row.get('CellType', '').lower().strip(),
                        'conflict_resolution': 'NO_CONFLICT',
                        'discarded_state': None,
                        'resolved_by': None,
                        'location_score': 0,
                        'speed': 0,
                        'vehicle_type': 'UNKNOWN',
                        'distance': 0,
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
                'start_time_str', 'end_time_str', 'state', 'confidence',
                'latitude', 'longitude', 'duration', 'tower_jump',
                'same_time_diff_state', 'cell_types', 'location_score',
                'speed_str', 'distance_str', 'vehicle_type', 'is_location_change_possible',
                'conflict_resolution', 'discarded_state', 'resolved_by'
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
    output_file_name = 'main_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
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