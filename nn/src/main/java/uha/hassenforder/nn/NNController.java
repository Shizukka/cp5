package uha.hassenforder.nn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.ws.rs.QueryParam;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserter;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

@RestController
public class NNController {

    @PostMapping (value="/config")
    @ResponseStatus(HttpStatus.OK)
    public String config (@QueryParam( "graph" ) String graph) {
        
        String url = "http://ia:80/config?graph={graph}";

        HttpHeaders headers = new HttpHeaders();
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(headers);
        
        Map<String, String> params = new TreeMap<>();
        params.put("graph", graph);

        RestTemplate template = new RestTemplate();
        ResponseEntity<String> response = template.exchange(url, HttpMethod.POST, requestEntity, String.class, params);

        return response.getBody();
    }

    @PostMapping (value="/classify")
    @ResponseStatus(HttpStatus.OK)
    public String classify (@RequestParam( "picture" ) MultipartFile picture) {

        try {
            // Lecture du contenu
            BufferedReader reader = new BufferedReader(new InputStreamReader(picture.getInputStream(), StandardCharsets.UTF_8));
            CSVParser csvParser = CSVFormat.TDF.parse(reader);

            // Récupération de la première ligne
            CSVRecord firstRecord = csvParser.getRecords().get(0);

            List<Double> doubles = new ArrayList<>();
            for (String value : firstRecord) {
                doubles.add(Double.parseDouble(value));
            }

            while (doubles.size() > 96) {
                doubles.remove(0);
            }

            DataTransfertObject requestBody = new DataTransfertObject();
            requestBody.setX_data(doubles);

            String url = "http://ia:80/evaluate";

            WebClient client = WebClient.create(url);

            ResponseTransfertObject response = client.post()
                    .uri("")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(BodyInserters.fromValue(requestBody))
                    .retrieve()
                    .bodyToMono(ResponseTransfertObject.class)
                    .block();

            return response.toString();
        } catch (IOException e) {
            e.printStackTrace();
            throw new HttpServerErrorException(HttpStatus.NOT_ACCEPTABLE);
        }
    }
}
