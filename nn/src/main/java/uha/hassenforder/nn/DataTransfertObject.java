package uha.hassenforder.nn;

import java.util.List;

public class DataTransfertObject {
    private List<Double> x_data;

    public List<Double> getX_data() {
        return x_data;
    }

    public void setX_data(List<Double> x_data) {
        this.x_data = x_data;
    }

    @Override
    public String toString() {
        return "DataTransfertObject{" +
                "x_data=" + x_data +
                '}';
    }
}
