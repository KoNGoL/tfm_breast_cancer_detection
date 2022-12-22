$(document).ready(function (e) {

    $("#submit_button").on("click", function () {
        blockUI(null, 'Procesando imagen, por favor espere.')
        $('#images_div').hide();
        var file_data = $("#upload").prop("files")[0];
        var form_data = new FormData();
        form_data.append("file", file_data);
        $.ajax({
        url: "/process_diccom",
        dataType: "text",
        cache: false,
        contentType: false,
        processData: false,
        data: form_data,
        type: "post",
        success: function (response) {
            imgs = JSON.parse(response)
            $('#yolo_div img').remove()
            $('#retinanet_div img').remove()
            $('#yolo_div').append('<img src="data:image/png;base64, ' + imgs.yolo + '" alt="Red dot" />')
            $('#retinanet_div').append('<img src="data:image/png;base64, ' + imgs.retinanet + '" alt="Red dot" />')
            $('#images_div').show();
            unblockUI();
        },
        error: function (response) {
            unblockUI();
            alert('Se ha producido un error al procesar la imagen.')
        },
        });
    });
});


function blockUI(html, text) {
    html = html || '';
    text = text || '';
    $.blockUI({
        message: '<p style="margin-bottom:5px">' + text + '</p>' + html,
        css: {
            border: 'none',
            padding: '15px',
            backgroundColor: '#000',
            'border-radius': '10px',
            '-webkit-border-radius': '10px',
            '-moz-border-radius': '10px',
            '-o-border-radius': '10px',
            '-ms-border-radius':' 10px',
            opacity: .5,
            color: '#fff'
        }
    });
}


/**
 * Desbloquea la GUI
 */
function unblockUI() {
    $.unblockUI();
}
