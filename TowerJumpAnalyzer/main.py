import csv
import json
import os
import re
from datetime import datetime
from shapely.geometry import Point, shape

class TowerJumpAnalyzer:
    def __init__(self):
        # Configurações principais
        self.min_time_diff_threshold = 5 * 60  # 5 minutos em segundos
        self.max_time_diff_threshold = 60 * 60  # 1 hora em segundos
        self.max_velocity_threshold = 900  # max km/h para considerar um tower jump
        self.max_jump_distance = 300  # Distância máxima para considerar um tower jump (em km)
        self.min_jump_distance = 10  # Distância mínima para considerar um tower jump (em km)

        self.date_format = '%m/%d/%y %H:%M'
        self.min_confidence = 0
        self.state_pattern = re.compile(r',\s*([A-Z]{2}|[A-Za-z\s]+)$')
        self.us_state_abbreviations = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 
            'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 
            'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
            'MD': 'Maryland', 'MA': 'Massachusetts', 
            'MI': 'Michigan', 'MN': 'Minnesota', 
            'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota',
            'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
            'PA': 'Pennsylvania', 'RI': 'Rhode Island',
            'SC': 'South Carolina', 'SD': 'South Dakota',
            'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 
            'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        
        # Carrega os polígonos dos estados (geojson simplificado)
        self.load_state_geometries()
        
        # Prioridade de tipos de célula
        self.cell_type_priority = {
            'macro': 1,
            'small': 2,
            'micro': 3,
            'pico': 4,
            'femto': 5,
            'unknown': 0
        }

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
        
        # Adiciona pontos por tipo de célula
        cell_type = entry['cell_types'].lower().strip()
        if cell_type in self.cell_type_priority:
            score += (100 - self.cell_type_priority[cell_type] * 10)

        # Penaliza se o estado for desconhecido
        if entry['state'] == 'UNKNOWN':
            score -= 20

        # Penaliza se a localização for pontual (sem coordenadas)
        if entry['latitude'] is None or entry['longitude'] is None:
            score -= 30

        # Penaliza se o registro for um tower jump
        if entry.get('tower_jump', False):
            score -= 50
        
        # Garante que o score não fique negativo
        score = max(score, 0)

        # Limita o score máximo a 100
        return min(score, 100)
        
    def load_state_geometries(self):
        """Carrega os polígonos dos estados a partir de um GeoJSON simplificado"""
        try:
            # GeoJSON simplificado com os polígonos dos estados
            states_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "New York"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-79.8, 40.5], [-71.8, 40.5], [-71.8, 45.1], [-79.8, 45.1], [-79.8, 40.5]]]
                        }
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Connecticut"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-73.7, 40.9], [-71.8, 40.9], [-71.8, 42.1], [-73.7, 42.1], [-73.7, 40.9]]]
                        }
                    },
                    # Adicione outros estados conforme necessário
                ]
            }
            
            self.state_geometries = {}
            for feature in states_geojson['features']:
                state_name = feature['properties']['name']
                self.state_geometries[state_name] = shape(feature['geometry'])
                
        except Exception as e:
            print(f"Erro ao carregar geometrias dos estados: {str(e)}")
            self.fallback_state_detection = True
        else:
            self.fallback_state_detection = False
    
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
    
    def resolve_state_conflicts(self, data):
        """Resolve conflitos de estado entre registros"""
        if not data:
            return data
        
        resolved_data = []
        for i in range(len(data)):
            current = data[i]
            if i == 0 or current['state'] != data[i - 1]['state']:
                resolved_data.append(current)
                continue
            
            previous = data[i - 1]
            time_diff = (current['start_time'] - previous['end_time']).total_seconds() / 60
            
            # Verifica se o tempo de diferença é significativo
            if time_diff < self.min_duration_to_override:
                # Resolve conflito com base na confiança
                if (current['confidence'] >= self.min_confidence_absolute and 
                    current['confidence'] > previous['confidence'] + self.min_confidence_diff):
                    current['conflict_resolution'] = 'OVERRIDE'
                    current['discarded_state'] = previous['state']
                    current['resolved_by'] = 'STATE_CONFLICT_RESOLUTION'
                else:
                    current['conflict_resolution'] = 'NO_CONFLICT'
                    current['discarded_state'] = None
                    current['resolved_by'] = None
            else:
                current['conflict_resolution'] = 'NO_CONFLICT'
                current['discarded_state'] = None
                current['resolved_by'] = None
            
            resolved_data.append(current)
        
        return resolved_data
    
    def calculate_velocity(self, point1, point2, time_diff):
        """Calcula a velocidade entre dois pontos em km/h"""
        if point1 is None or point2 is None or time_diff <= 0:
            return 0
        
        distance = point1.distance(point2) * 111  # Distância em km
        velocity = distance / (time_diff / 3600)  # km/h
        return velocity
    
    def detect_vehicle_type(self, velocity, distance, time_diff):
        """Detecta o tipo de veículo com base na velocidade, tempo e distância"""
        if velocity <= 0:
            return 'UNKNOWN'
        
        vehicle_types = {
            'Avião': (200, 900),
            'Carro': (20, 200),
            'Barco': (5, 100),
            'Trem': (30, 300),
            'Ônibus': (10, 120)
        }
        
        for vehicle, (min_speed, max_speed) in vehicle_types.items():
            if min_speed <= velocity <= max_speed:
                if distance > 0 and time_diff > 0:
                    # Verifica se a velocidade é compatível com a distância e o tempo
                    # através do calculo de distância percorrida
                    if (velocity * (time_diff / 3600)) >= distance:
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
        objeto['velocity'] = data.get('velocity', 0)
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

        # previous['tower_jump'] = True
        # previous['discarded_state'] = previous['state'] 
        # previous['resolved_by'] = 'TOWER_JUMP_DETECTION'
        return current

    def detect_jumps(self, data):
        """Detecta tower jumps com base nos dados de localização"""
        if not data:
            return data
        
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i - 1]
            
            if not current['start_time'] or not previous['start_time']:
                continue

            if current['latitude'] is None or current['longitude'] is None:
                continue

            time_diff = (current['start_time'] - previous['end_time']).total_seconds()

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
                previous['page'] = data[i].get('page', '')
                previous['item'] = data[i].get('item', '')
            
            current_state = current['state']

            if previous_state == current_state:
                if previous['start_time'] == current['start_time']:
                    current['same_time_diff_state'] = 'SAME_TIME_DIFF_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFFERENT_TIME_DIFF_STATE'

            if previous['latitude'] is  None and previous['longitude'] is None or \
                previous['latitude'] == 0 or previous['longitude'] == 0:
                previous = self.criar_objeto_padrao(previous)
                continue
                
            if current['latitude'] is None or current['longitude'] is None or \
                current['latitude'] == 0 or current['longitude'] == 0:
                current = self.criar_objeto_padrao(current)
                continue

            if current_state == 'UNKNOWN':
                current = self.criar_objeto_padrao(current)
                continue

            current_point = Point(current['longitude'], current['latitude'])
            previous_point = Point(previous['longitude'], previous['latitude'])
            
            velocity = self.calculate_velocity(
                previous_point,
                current_point,
                time_diff
            )
            
            current['vehicle_type'] = self.detect_vehicle_type(velocity, current.get('distance', 0), time_diff)
            current['velocity'] = velocity
            previous['velocity'] = previous.get('velocity', 0)

            current['location_score'] = self.calculate_location_score(current)
            previous['location_score'] = self.calculate_location_score(previous)

            current['same_time_diff_state'] = (current['state'] == previous['state'])
            current['duration'] = (current['start_time'] - previous['end_time']).total_seconds() / 60

            distance = current_point.distance(previous_point) * 111  # Distância em km
            current['distance'] = distance

            is_velocyty_high = velocity > self.max_velocity_threshold

            if is_velocyty_high:
                current['conflict_resolution'] = 'HIGH_VELOCITY'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # Distância incompatível com o tempo:
            if current['state'] != previous['state'] and current['vehicle_type'] == 'UNKNOWN':
                current['conflict_resolution'] = 'INCOMPATIBLE_DISTANCE'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # Lógica de detecção de tower jump
            # 1. Registros simultâneos em estados diferentes:
            if (current['state'] != previous['state'] and 
                current['same_time_diff_state'] and 
                time_diff < self.min_time_diff_threshold):
                current['conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # 2. Sequência rápida de estados diferentes:
            if (current['state'] != previous['state'] and 
                current['same_time_diff_state'] and 
                time_diff < self.max_time_diff_threshold):
                
                # Verifica se a velocidade é alta ou se a distância é grande
                if (is_velocyty_high and distance > self.min_jump_distance) or \
                    (velocity > 0 and time_diff < self.min_time_diff_threshold and distance > self.min_jump_distance):
                    # Marca como tower jump se a velocidade for alta ou a distância for grande  
                    current['conflict_resolution'] = 'RAPID_STATE_JUMP'
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue

            # Registros intercalados de estados diferentes:
            if (current['state'] != previous['state'] and
                not current['same_time_diff_state']):
                # Verifica se a velocidade é alta ou se a distância é grande
                if (is_velocyty_high and distance > self.min_jump_distance) or \
                    (velocity > 0 and time_diff < self.min_time_diff_threshold and distance > self.min_jump_distance):
                    # Marca como tower jump se a velocidade for alta ou a distância for grande  
                    current['conflict_resolution'] = 'INTERLEAVED_STATE_JUMP'
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue

            # Verifica se o tempo de diferença é curto e o estado é igual (N tower jumps consecutivos para o mesmo estado):
            if current['state'] == previous['state'] and current['vehicle_type'] != 'UNKNOWN':
                # Marca como tower jump se a velocidade for alta ou a distância for grande  
                current['conflict_resolution'] = 'CONSECUTIVE_STATE_JUMP'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # 3. Registros simultâneos em estados diferentes:
            if (current['state'] != previous['state'] and 
                current['same_time_diff_state']):
                current['conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # if time_diff > self.min_time_diff_threshold and distance > self.min_jump_distance:
            #     current['conflict_resolution'] = 'LONG_DISTANCE_JUMP'
            #     current = self.marcar_tower_jump(current, valid_previous_state)
            #     continue
            
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
                    has_header = True
                
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
                        
                        lat = self.safe_float(row.get('Latitude'))
                        lon = self.safe_float(row.get('Longitude'))
                        
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
                            'location_score': 0
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
                    'conflict_resolution', 'discarded_state', 'resolved_by'
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
                        'conflict_resolution': entry['conflict_resolution'],
                        'discarded_state': entry['discarded_state'] or '',
                        'resolved_by': entry['resolved_by'] or ''
                    })
            return True
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
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
    output_file = os.path.join(base_dir, "TowerJumpReport.csv")

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