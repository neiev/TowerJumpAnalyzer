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
        """Converte string para float, retornando None se falhar"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def parse_datetime(self, datetime_str):
        """Converte string de data/hora para objeto datetime"""
        try:
            return datetime.strptime(datetime_str, self.date_format)
        except ValueError as e:
            print(f"Erro ao analisar data/hora: {str(e)}")
            return None
        
    def calculate_location_score(self, entry):
        """Calcula o score de localização baseado em confiança e tipo de célula"""
        score = 0
        if entry['confidence'] >= self.min_confidence:
            score += entry['confidence']

        # Penaliza se o estado for desconhecido
        if entry['state'] == 'UNKNOWN':
            score -= 20

        # Penaliza se a localização for pontual (sem coordenadas)
        if entry['latitude'] is None or entry['longitude'] is None or \
            (entry['latitude'] == 0 and entry['longitude'] == 0):
            score -= 30

        # Penaliza se o registro for um tower jump
        if entry.get('tower_jump', False):
            score -= 50

        # Garante que o score não fique negativo
        score = max(score, 0)

        # Limita o score máximo a 100
        return min(score, 100)
    
    def get_current_location(self, data):
        """Obtém a localização atual com base nos dados mais recentes"""
        if not data:
            return {'status': 'NO_DATA', 'state': 'UNKNOWN', 'coordinates': None, 'confidence': 0, 'location_quality': 0}
        
        # Ordena por timestamp e pega o mais recente
        latest_entry = max(data, key=lambda x: x['start_time'])
        
        if latest_entry['state'] == 'UNKNOWN':
            return {
                'status': 'NO_VALID_DATA',
                'state': latest_entry['state'],
                'coordinates': (latest_entry['latitude'], latest_entry['longitude']),
                'confidence': latest_entry['confidence'],
                'location_quality': latest_entry['location_score'],
                'location_score': latest_entry['location_score'],
                'last_update': latest_entry['start_time'],
                'minutes_since_update': 0,
                'tower_jump': latest_entry.get('tower_jump', False),
                'cell_type': latest_entry.get('cell_types', '')
            }
        
        return {
            'status': 'CURRENT',
            'state': latest_entry['state'],
            'coordinates': (latest_entry['latitude'], latest_entry['longitude']),
            'confidence': latest_entry['confidence'],
            'location_quality': latest_entry['location_score'],
            'location_score': latest_entry['location_score'],
            'last_update': latest_entry['start_time'],
            'minutes_since_update': (latest_entry['start_time'] - data[0]['start_time']).total_seconds() / 60,
            'tower_jump': latest_entry.get('tower_jump', False),
            'cell_type': latest_entry.get('cell_types', '')
        }
    
    def calcular_distancia_e_velocidade(self, lat1_deg, lon1_deg, lat2_deg, lon2_deg, tempo_segundos):
        """Calcula a distância em km e velocidade média em km/h entre dois pontos.
        
        Args:
            lat1_deg: Latitude do ponto 1 em graus
            lon1_deg: Longitude do ponto 1 em graus
            lat2_deg: Latitude do ponto 2 em graus
            lon2_deg: Longitude do ponto 2 em graus
            tempo_segundos: Tempo decorrido em segundos
            
        Returns:
            tuple: (distância_km, velocidade_kmh)
        """
        # Verificação rápida para pontos idênticos
        if (lat1_deg, lon1_deg) == (lat2_deg, lon2_deg):
            return 0.0, 0.0
        
        R = 6371  # Raio da Terra em km
        
        # Converter graus para radianos
        lat1_rad = math.radians(lat1_deg)
        lon1_rad = math.radians(lon1_deg)
        lat2_rad = math.radians(lat2_deg)
        lon2_rad = math.radians(lon2_deg)

        # Cálculos intermediários
        sin_lat1 = math.sin(lat1_rad)
        cos_lat1 = math.cos(lat1_rad)
        sin_lat2 = math.sin(lat2_rad)
        cos_lat2 = math.cos(lat2_rad)
        delta_lon = lon2_rad - lon1_rad
        cos_delta_lon = math.cos(delta_lon)

        # Cálculo do ângulo central com proteção contra overflow
        cos_angle = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lon
        
        # Normalizar para o intervalo [-1, 1]
        cos_angle = max(min(cos_angle, 1.0), -1.0)

        # Calcular distância
        d = R * math.acos(cos_angle)  # em km

        # Tratar tempo zero ou negativo
        if tempo_segundos <= 0:
            return d, 0.0

        # Calcular velocidade (km/h)
        velocidade_kmh = (d * 3600) / tempo_segundos
        
        return d, velocidade_kmh

    
    def detect_vehicle_type(self, speed, distance, time_diff):
        """Detecta o tipo de veículo com base na velocidade, tempo e distância"""
        if speed <= 0:
            return 'UNKNOWN'
        
        vehicle_types = {
            'A Pé': (0, 7),
            'Bicicleta': (7, 30),
            'Carro': (20, 200),
            'Ônibus': (10, 120),
            'Barco': (5, 100),
            'Trem': (30, 300),
            'Avião': (200, 900),
        }
        
        for vehicle, (min_speed, max_speed) in vehicle_types.items():
            if min_speed <= speed <= max_speed:
                if distance > 0 and time_diff > 0:
                    # Verifica se a velocidade é compatível com a distância e o tempo
                    # através do calculo de distância percorrida
                    if (speed * (time_diff / 3600)) >= distance:
                        return vehicle
                    
        return 'UNKNOWN'
    
    def criar_objeto_padrao(self, data):
        objeto = {}
        
        objeto['latitude'] = data['latitude']
        objeto['longitude'] = data['longitude']
        objeto['start_time'] = data['start_time']
        objeto['end_time'] = data['end_time']
        objeto['vehicle_type'] = data.get('vehicle_type', 'UNKNOWN')
        objeto['location_score'] = data.get('location_score', 0)
        objeto['cell_types'] = data.get('cell_types', '')
        objeto['confidence'] = data.get('confidence', 0)
        objeto['tower_jump'] = data.get('tower_jump', False)
        objeto['conflict_resolution'] = data.get('conflict_resolution', 'NO_CONFLICT')
        objeto['discarded_state'] = data.get('discarded_state', None)
        objeto['resolved_by'] = data.get('resolved_by', None)
        objeto['distance'] = data.get('distance', 0)
        objeto['speed'] = data.get('speed', 0)
        objeto['is_location_change_possible'] = data.get('is_location_change_possible', False)
        objeto['duration'] = data.get('duration', 0)
        objeto['same_time_diff_state'] = data.get('same_time_diff_state', 'DIFFERENT_TIME_DIFF_STATE')
        objeto['state'] = data.get('state', 'UNKNOWN')
        objeto['page'] = data.get('page', '')
        objeto['item'] = data.get('item', '')
        
        return objeto

    def marcar_tower_jump(self, current, valid_previous_state):
        current['tower_jump'] = True
        current['discarded_state'] = valid_previous_state
        current['resolved_by'] = 'TOWER_JUMP_DETECTION'
        return current
    
    def is_valid_record(self, record):
        """Verifica se o registro é válido para análise"""
        if not record:
            return False
        
        if 'start_time' not in record or 'end_time' not in record:
            return False
        
        if not isinstance(record['start_time'], datetime) or not isinstance(record['end_time'], datetime):
            return False
        
        if record['start_time'] >= record['end_time']:
            return False
        
        if 'latitude' not in record or 'longitude' not in record:
            return False
        
        if record['latitude'] is None or record['longitude'] is None:
            return False
        
        if record['latitude'] == 0 and record['longitude'] == 0:
            return False
        
        if record['latitude'] <= -90 and record['longitude'] <= -180:
            return False

        return True
    
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

    def detect_jumps(self, data):
        """Detecta tower jumps com base nos dados de localização"""
        if not data:
            return data
        
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i - 1]
            
            if not current['start_time'] or not previous['start_time']:
                continue

            if not current['state'] or \
                current['state'] == '' or \
                current['state'] == 'UNKNOWN' or \
                current['state'] is None:
                # se não tem estado válido, cria um objeto padrão
                current = self.criar_objeto_padrao(current)
                continue

            if (current['latitude'] is None or current['longitude'] is None or \
                current['latitude'] == 0 or current['longitude'] == 0):
                current = self.criar_objeto_padrao(current)
                continue

            if 'tower_jump' not in previous:
                previous['tower_jump'] = False

            previous_state = previous['state']

            valid_previous_state = 'UNKNOWN'
            if previous_state == 'UNKNOWN':
                while i > 1 and valid_previous_state == 'UNKNOWN':
                    i -= 1
                    if data[i]['state'] and data[i]['state'] != 'UNKNOWN':
                        valid_previous_state = data[i]['state']      
                if valid_previous_state == 'UNKNOWN':
                    continue

                previous['state'] = valid_previous_state
                previous['latitude'] = data[i]['latitude']
                previous['longitude'] = data[i]['longitude']
                previous['start_time'] = data[i]['start_time']
                previous['end_time'] = data[i]['end_time']
                previous['tower_jump'] = data[i]['tower_jump']

            if valid_previous_state == current['state'] and \
                current['state'] != 'UNKNOWN':
                if previous['start_time'] == current['start_time']:
                    current['same_time_diff_state'] = 'SAME_TIME_SAME_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFF_TIME_SAME_STATE'
            else:
                if previous['start_time'] == current['start_time']:
                    current['same_time_diff_state'] = 'SAME_TIME_DIFF_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFF_TIME_DIFF_STATE'

            time_diff = (current['start_time'] - previous['end_time']).total_seconds()

            # current_point = Point(current['longitude'], current['latitude'])
            # previous_point = Point(previous['longitude'], previous['latitude'])

            # current['distance'] = previous_point.distance(current_point) * 111  # Distância em km

            distance, speed = self.calcular_distancia_e_velocidade(current['longitude'], 
                                                                   current['latitude'],
                                                                   previous['longitude'], 
                                                                   previous['latitude'], 
                                                                   time_diff)
            current['distance'] = distance
            current['speed'] = speed
            previous['speed'] = previous.get('speed', 0)
            current['vehicle_type'] = self.detect_vehicle_type(current['speed'], current.get('distance', 0), time_diff)
        
            current['location_score'] = self.calculate_location_score(current)
            previous['location_score'] = self.calculate_location_score(previous)

            current['duration'] = (current['start_time'] - previous['end_time']).total_seconds() / 60

            # Verifica se a velocidade é muito alta
            if current['speed'] > self.max_speed_threshold:
                current['conflict_resolution'] = 'HIGH_speed'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            is_same_state = current['state'] == previous['state']

            if is_same_state and \
                current['same_time_diff_state'] == 'SAME_TIME_SAME_STATE':
                current['conflict_resolution'] = 'SAME_STATE_SAME_TIME'
                if previous['tower_jump']:
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue

            elif is_same_state and \
                current['same_time_diff_state'] == 'DIFF_TIME_SAME_STATE':
                current['conflict_resolution'] = 'SAME_STATE_DIFF_TIME'
                if previous['tower_jump'] and time_diff < self.min_time_diff_threshold:
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue
            elif not is_same_state and \
                current['same_time_diff_state'] == 'SAME_TIME_DIFF_STATE':
                current['conflict_resolution'] = 'DIFF_STATE_SAME_TIME'
                # checa o estado válido para o intervalo de tempo de registro
                is_consistent = self.check_consistency(current, data, i)    
                if not is_consistent:
                    current['conflict_resolution'] = 'DIFF_STATE_JUMP'
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue


            # Verifica se o deslocamento é possível
            is_location_change_possible = False
            # calcule com base na distância e tempo e velocidade e tipo de veículo
            # não usa apenas o minimo e máximo e sim faz cáluclo matemático real de deslocamento
            if current['distance'] > 0 and time_diff > 0:
                is_location_change_possible = (current['speed'] * (time_diff / 3600)) >= current['distance']
            current['is_location_change_possible'] = is_location_change_possible

            # Verifica se a velocidade é impossível
            if (not is_location_change_possible):
                current['conflict_resolution'] = 'LOCATION_CHANGE_IMPOSSIBLE'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # 1. Registros simultâneos em estados diferentes:
            if (not is_same_state and 
                not current['same_time_diff_state'] == 'SAME_TIME_DIFF_STATE'):
                current['conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
                # checa o estado válido para o intervalo de tempo de registro
                is_consistent = self.check_consistency(current, data, i)
                if not is_consistent:
                    current['conflict_resolution'] = 'DIFF_STATE_JUMP'
                    current = self.marcar_tower_jump(current, valid_previous_state)
                continue
            # 2. Registros com estado diferente e tempo de diferença curto
            elif (not is_same_state and 
                  current['same_time_diff_state'] == 'DIFF_TIME_DIFF_STATE' and 
                  time_diff < self.min_time_diff_threshold):
                current['conflict_resolution'] = 'DIFF_STATE_SHORT_TIME'
                # checa o estado válido para o intervalo de tempo de registro
                is_consistent = self.check_consistency(current, data, i)
                if not is_consistent and \
                   not current['is_location_change_possible']:
                    current = self.marcar_tower_jump(current, valid_previous_state)
                continue
            
        return data

    def load_data(self, file_path):
        """Carrega e processa os dados do arquivo CSV"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                try:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                    has_header = csv.Sniffer().has_header(f.read(1024))
                    f.seek(0)
                except:
                    dialect = 'excel'
                
                reader = csv.DictReader(f, dialect=dialect)
                
                required_columns = {'UTCDateTime', 'LocalDateTime', 'State', 'Latitude', 'Longitude'}
                if not all(col in reader.fieldnames for col in required_columns):
                    missing = required_columns - set(reader.fieldnames)
                    print(f"Erro: Colunas obrigatórias faltando: {missing}")
                    return []
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        datetime_str = row.get('UTCDateTime') or row.get('LocalDateTime')
                        start_time = self.parse_datetime(datetime_str)
                        
                        if not start_time:
                            print(f"Aviso: Ignorando linha {row_num} - formato de data/hora inválido: {datetime_str}")
                            continue

                        end_time = start_time
                        
                        unfrormated_lat = row.get('Latitude')
                        if unfrormated_lat and isinstance(unfrormated_lat, str):
                            unfrormated_lat = unfrormated_lat[:8]
                        lat = unfrormated_lat

                        unfrormated_lon = row.get('Longitude')
                        if unfrormated_lon and isinstance(unfrormated_lon, str):
                            unfrormated_lon = unfrormated_lon[:9]
                        lon = unfrormated_lon

                        lat = self.safe_float(lat)
                        lon = self.safe_float(lon)
                        
                        # Determina o estado - prioridade para coordenadas
                        state = 'UNKNOWN'
                        
                        if lat is not None and lon is not None:
                            state = row.get('State', '')
                        
                        # Cria a entrada de dados
                        entry = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'state': state,
                            'confidence': 100,  # Assume alta confiança para dados com coordenadas
                            'latitude': lat,
                            'longitude': lon,
                            'duration': 0,
                            'tower_jump': False,
                            'same_time_diff_state': False,
                            'cell_types': row.get('CellType', '').lower().strip(),
                            'raw_data': row,
                            'conflict_resolution': 'NO_CONFLICT',
                            'discarded_state': None,
                            'resolved_by': None,
                            'location_score': 0,
                            'speed': 0,
                            'vehicle_type': 'UNKNOWN',
                            'distance': 0,
                        }
                        
                        # entry['location_score'] = self.calculate_location_score(entry)
                        data.append(entry)
                        
                    except Exception as e:
                        print(f"Erro ao processar linha {row_num}: {str(e)}")
                        continue
                    
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {file_path}")
            return []
        except Exception as e:
            print(f"Erro inesperado ao ler arquivo: {str(e)}")
            return []

        if not data:
            print("Aviso: O arquivo foi lido, mas nenhum dado válido foi encontrado")
            return []

        data = sorted(data, key=lambda x: x['start_time'])
        print(f"Carregados {len(data)} registros válidos")
        
        unknown_count = sum(1 for e in data if e['state'] == 'UNKNOWN')
        if unknown_count > 0:
            print(f"AVISO: {unknown_count} registros ({unknown_count/len(data):.2%}) com estado UNKNOWN")
            if unknown_count/len(data) > 0.05:
                print("  Ação recomendada: Verificar qualidade das coordenadas e mapeamento de estados")
        
        return data
    
    def format_distance(self, distance):
        """Formata a distância em km com 2 casas decimais"""
        try:
            if distance is None:
                return ''
            return f"{distance:.2f} km" if distance >= 0 else ''
        except (ValueError, TypeError) as e:
            print(f"Erro ao formatar distância: {str(e)}")
            print(distance)
            traceback.print_exc()

    
    def format_speed(self, speed):
        """Formata a velocidade em km/h com 2 casas decimais"""
        if speed is None:
            return ''
        return f"{speed:.2f} km/h" if speed >= 0 else ''

    def generate_report(self, data, output_file):
        """Gera um relatório CSV com os dados processados"""
        if not data:
            print("Nenhum dado para gerar relatório")
            return False
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'start_time', 'end_time', 'state', 'confidence', 
                    'latitude', 'longitude', 'duration', 'tower_jump',
                    'same_time_diff_state', 'cell_types', 'location_score',
                    'speed', 'distance',  'vehicle_type', 'is_location_change_possible',
                    'conflict_resolution', 'discarded_state', 'resolved_by', 
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in data:
                    writer.writerow({
                        'start_time': entry['start_time'].strftime(self.date_format),
                        'end_time': entry['end_time'].strftime(self.date_format),
                        'state': entry['state'],
                        'confidence': entry['confidence'],
                        'latitude': entry['latitude'],
                        'longitude': entry['longitude'],
                        'duration': entry['duration'],
                        'tower_jump': entry['tower_jump'],
                        'same_time_diff_state': entry['same_time_diff_state'],
                        'cell_types': entry['cell_types'],
                        'location_score': entry['location_score'],
                        'speed': self.format_speed(entry['speed']),
                        'distance': self.format_distance(entry['distance']),
                        'vehicle_type': entry['vehicle_type'] or 'UNKNOWN',
                        'is_location_change_possible': entry.get('is_location_change_possible', False),
                        'conflict_resolution': entry['conflict_resolution'],
                        'discarded_state': entry['discarded_state'] or '',
                        'resolved_by': entry['resolved_by'] or ''
                    })
            return True
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            traceback.print_exc()
            return False
        
    def print_summary(self, data):
        """Imprime um resumo dos dados processados"""
        if not data:
            print("Nenhum dado para resumo")
            return
        
        total_records = len(data)
        tower_jump_count = sum(1 for e in data if e['tower_jump'])
        unknown_state_count = sum(1 for e in data if e['state'] == 'UNKNOWN')
        
        print("\n=== RESUMO DOS DADOS ===")
        print(f"Total de registros: {total_records}")
        print(f"Registros com Tower Jump: {tower_jump_count} ({tower_jump_count/total_records:.2%})")
        print(f"Registros com estado UNKNOWN: {unknown_state_count} ({unknown_state_count/total_records:.2%})")

def main():
    print("=== Tower Jump Analyzer ===")
    print("Sistema avançado de análise de localização e detecção de tower jumps\n")

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
    data = analyzer.load_data(input_file)

    if not data:
        print("Nenhum dado válido encontrado no arquivo de entrada.")
        return

    print("Detectando tower jumps...")
    data = analyzer.detect_jumps(data)

    # print("Resolvendo conflitos de estado...")
    # data = analyzer.resolve_state_conflicts(data)

    # print("Obtendo localização atual...")
    # current_location = analyzer.get_current_location(data)

    print("Gerando relatório...")
    if analyzer.generate_report(data, output_file):
        print(f"\nRelatório gerado com sucesso em: {output_file}")
    else:
        print("Falha ao gerar relatório.")
        return

    analyzer.print_summary(data)

if __name__ == "__main__":
    main()