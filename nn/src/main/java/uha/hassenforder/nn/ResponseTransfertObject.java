package uha.hassenforder.nn;

public class ResponseTransfertObject {
    private Double result;

    private String status;

    public Double getResult() {
        return result;
    }

    public void setResult(Double result) {
        this.result = result;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    @Override
    public String toString() {
        return "ResponseTransfertObject{" +
                "result=" + result +
                ", status='" + status + '\'' +
                '}';
    }
}
