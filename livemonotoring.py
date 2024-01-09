import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

class LivePlotter:
    def __init__(self):
        # Create application
        self.app = QApplication(sys.argv)

        # Create main window and graph widget
        self.main_window = pg.GraphicsLayoutWidget(show=True, title="Live Heart Rate Monitor")
        self.plot = self.main_window.addPlot(title="Heart Rate")
        self.plot.setYRange(50, 120, padding=0)  # Adjust based on expected BPM range
        self.plot.setBackground('k')
        self.curve = self.plot.plot(pen=pg.mkPen(color='g', width=2))

        # Initialize data arrays
        self.time_data = np.linspace(0, 10, 1000)  # Initial time array
        self.heart_rate_data = np.random.randint(60, 100, size=1000)  # Initial heart rate data

        # Set up a timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)  # Update every 20ms

    def simulate_heart_rate(self):
        """Simulate receiving new heart rate data."""
        return np.random.randint(60, 100)

    def update(self):
        # Update data
        self.heart_rate_data[:-1] = self.heart_rate_data[1:]  # Shift data
        self.heart_rate_data[-1] = self.simulate_heart_rate()
        self.time_data[:-1] = self.time_data[1:]  # Shift time
        self.time_data[-1] += 0.02  # Increment last time

        # Update the plot
        self.curve.setData(self.time_data, self.heart_rate_data)

    def start(self):
        # Start the application
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    plotter = LivePlotter()
    plotter.start()