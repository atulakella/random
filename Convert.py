import os
import csv

# Define paths
base_path = r'D:\Projects\shrimpIQ\Imgs+labels'
class_names = ['Good', 'Damaged']

# Helper function to convert YOLO bounding boxes to TensorFlow format
def yolo_to_tf_format(label_file_path, img_width, img_height):
    boxes = []
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert to pixel coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            # Convert to TensorFlow format: xmin, ymin, xmax, ymax
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            boxes.append((xmin, ymin, xmax, ymax))
    
    return boxes

def create_csv_file(output_csv_path, dataset_type):
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Type', 'File Location', 'Class Name', 'Coordinates'])

        # Iterate over class names and their corresponding folders
        for class_name in class_names:
            img_folder = os.path.join(base_path, class_name, 'images')
            label_folder = os.path.join(base_path, class_name, 'labels')
            
            # Process each image in the folder
            for img_filename in os.listdir(img_folder):
                img_file_path = os.path.join(img_folder, img_filename)
                label_file_path = os.path.join(label_folder, img_filename.replace('.jpg', '.txt'))
                
                if not os.path.exists(label_file_path):
                    continue
                
                # Get image dimensions
                from PIL import Image
                with Image.open(img_file_path) as img:
                    img_width, img_height = img.size
                
                # Convert YOLO format labels to TensorFlow format
                boxes = yolo_to_tf_format(label_file_path, img_width, img_height)
                
                for box in boxes:
                    # Write to CSV file
                    csv_writer.writerow([dataset_type, img_file_path, class_name, ','.join(map(str, box))])

if __name__ == '__main__':
    # Output CSV file path
    output_csv_path = r'D:\Projects\shrimpIQ\Imgs+labels\dataset.csv'
    # Choose dataset type (e.g., 'TRAIN', 'TEST', 'VALIDATE')
    dataset_type = 'TRAIN'
    create_csv_file(output_csv_path, dataset_type)
